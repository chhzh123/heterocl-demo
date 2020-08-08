import heterocl as hcl
import hlib.op.bnn as bnn
import hlib.op.nn as nn
import numpy as np
import os, time, sys, argparse
import torch
import torchvision
import torchvision.transforms as transforms

target = None
batch_size = 100
qtype_bit = hcl.UInt(1) # weights
qtype_int = hcl.Int(6) # not unsigned!
qtype_float = hcl.Fixed(20,10)
qtype_packed = hcl.UInt(32)

def RSign(data, alpha, name="rsign", dtype=hcl.UInt(2)):
    assert data.shape[1] == alpha.shape[0]
    return hcl.compute(data.shape, lambda nn, cc, ww, hh: 
                        hcl.select(data[nn,cc,ww,hh] > alpha[cc], 1, 0),
                        name=name, dtype=dtype)

def RPReLU(data, x0, y0, beta, name="rprelu", dtype=None):
    assert data.shape[1] == beta.shape[0] \
        and x0.shape[0] == y0.shape[0] \
        and beta.shape[0] == x0.shape[0]
    dtype = data.dtype if dtype == None else dtype
    return hcl.compute(data.shape, lambda nn, cc, ww, hh:
                        hcl.select(data[nn,cc,ww,hh] > x0[cc],
                        data[nn,cc,ww,hh] - x0[cc] + y0[cc],
                        beta[cc] * (data[nn,cc,ww,hh] - x0[cc]) + y0[cc]),
                        name=name, dtype=dtype)

class BasicBlock():

    def __init__(self, in_planes, planes, stride, params, name="bb"):
        self.params = dict()
        self.params["rprelu1"] = params[:3]
        self.params["rprelu2"] = params[3:6]
        self.params["rsign1"] = params[6]
        self.params["rsign2"] = params[7]
        self.params["conv1"] = params[8]
        self.params["bn1"] = params[9:13] # ignore "num_batches_tracked"
        self.params["conv2"] = params[14]
        self.params["bn2"] = params[15:19]
        self.stride = stride
        self.flag = in_planes != planes
        self.name = name

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # 1st residual block
        rsign1 = RSign(x, self.params["rsign1"], name=self.name+"_rsign1")
        conv1 = bnn.conv2d_nchw(rsign1, self.params["conv1"], padding=[1,1], name=self.name+"_conv1") # no bias!
        bn1, _, _ = nn.batch_norm(conv1, *self.params["bn1"], name=self.name+"_bn1")
        if self.stride != 1 or self.flag:
            avgpool = nn.avg_pool2d_nchw(bn1, pooling=[2,2],
                                         stride=[2,2], padding=[0,0],
                                         name=self.name+"_avgpool")
            # dont use nn.concatenate!
            shape = avgpool.shape
            shortcut = hcl.compute((shape[0], shape[1]*2, shape[2], shape[3]),
                                    lambda nn, cc, ww, hh: avgpool[nn, cc % shape[1], ww, hh],
                                    name=self.name+"_concat")
        else:
            shortcut = x
        residual1 = hcl.compute(bn1.shape, lambda nn, cc, ww, hh:
                                bn1[nn, cc, ww, hh] + shortcut[nn, cc, ww, hh],
                                name=self.name+"_residual1")
        # 2nd residual block
        rprelu1 = RPReLU(residual1, *self.params["rprelu1"], name=self.name+"_rprelu1")
        rsign2 = RSign(rprelu1, self.params["rsign2"], name=self.name+"_rsign2")
        conv2 = bnn.conv2d_nchw(rsign2, self.params["conv2"], padding=[1,1], name=self.name+"_conv2")
        bn2, _, _ = nn.batch_norm(conv2, *self.params["bn2"], name=self.name+"_bn2")
        residual2 = hcl.compute(rprelu1.shape, lambda nn, cc, ww, hh:
                                bn2[nn, cc, ww, hh] + rprelu1[nn, cc, ww, hh],
                                name=self.name+"_residual2")
        rprelu2 = RPReLU(residual2, *self.params["rprelu2"], name=self.name+"_rprelu2")
        return rprelu2

class Sequential():

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class ResNet():

    def __init__(self, block, num_blocks, params):
        self.in_planes = 16
        self.params = dict()
        self.params["conv1"] = params[0]
        self.params["bn1"] = params[1:5] # ignore "num_batches_tracked"
        self.params["layer1"] = params[6:66]
        self.params["layer2"] = params[66:126]
        self.params["layer3"] = params[126:186]
        self.params["linear"] = params[186:]
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, params=self.params["layer1"], id=0)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, params=self.params["layer2"], id=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, params=self.params["layer3"], id=2)

    def _make_layer(self, block, planes, num_blocks, stride, params, id):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, params[i*20:(i+1)*20], name="bb{}".format(id*3+i)))
            self.in_planes = planes

        return Sequential(*layers)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        conv1 = nn.conv2d_nchw(x, self.params["conv1"], strides=[1, 1], padding=[1, 1], name="out_conv")
        layer1 = self.layer1(conv1)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        avgpool = nn.avg_pool2d_nchw(layer3, pooling=[2, 2], stride=[2, 2], padding=[0, 0], name="avgpool")
        flat = bnn.flatten(avgpool, name="flatten")
        out = nn.dense(flat, self.params["linear"][0], bias=self.params["linear"][1], name="fc")
        return out

def build_resnet20(*params): # params are placeholders here
    resnet = ResNet(BasicBlock, [3, 3, 3], params[1:])
    return resnet(params[0])

# declare hcl placeholders
def build_resnet20_inf(images, params, batch_size=batch_size, target=target):
    hcl_ph = []
    input_image = hcl.placeholder((batch_size,*images.shape[1:]),"input_image")
    for name in params:
        hcl_ph.append(hcl.placeholder(params[name].shape,name))

    # build the network
    scheme = hcl.create_scheme([input_image] + hcl_ph, build_resnet20)
    s = hcl.create_schedule_from_scheme(scheme)

    # if isinstance(target,hcl.platform):
    #     s.to([input_image] + hcl_ph, target.xcel)
    #     s.to(build_bnn.fc2, target.host)
    #     target.config(compile="vivado_hls", mode="csyn")

    return hcl.build(s, target=target)

def load_cifar10():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
        ])
    test_set = torchvision.datasets.CIFAR10(root='.', train=False,
                                           download=True, transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return test_set.data, np.array(test_set.targets)

params = torch.load("pretrained-models/model_react-resnet20.pt", map_location=torch.device("cpu"))
print("Loading the data.")
images, labels = load_cifar10()
num_images = len(images)
for key in params:
    if "rprelu" in key or "binarize" in key:
        params[key] = params[key].reshape(-1)

hls_code = build_resnet20_inf(images, params, target="vhls")
with open("vhls_code.cpp","w") as outfile:
    outfile.write(hls_code)
print("Finish generating Vivado HLS code.")

resnet20 = build_resnet20_inf(images, params)
print("Finish building function.")

hcl_array = []
for name in params:
    hcl_array.append(hcl.asarray(params[name],dtype=qtype_float))
hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)

correct_sum = 0
for i in range(num_images // batch_size):
    np_image = images[i*batch_size:(i+1)*batch_size]
    hcl_image = hcl.asarray(np_image, dtype=qtype_float)
    resnet20(hcl_image, *hcl_array, hcl_out)
    prediction = np.argmax(hcl_out.asnumpy(), axis=1)
    correct_sum += np.sum(np.equal(prediction, labels[i*batch_size:(i+1)*batch_size]))
    if (i+1) % 10 == 0:
        print("Done {} batches.".format(i+1))
print("Testing accuracy: {}".format(correct_sum / float(num_images)))
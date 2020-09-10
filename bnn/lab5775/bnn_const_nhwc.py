import heterocl as hcl
import hlib.op.bnn as bnn
import numpy as np
import os, time, sys, argparse
from heterocl.profiler import Profiler

profiler = Profiler()

parser = argparse.ArgumentParser()
parser.add_argument('--vitis', type=bool, default=False,
                    help='Use Vitis to compile? (default: False)')
parser.add_argument('--opt', type=bool, default=False,
                    help='Use optimization? (default: False)')
parser.add_argument('--stream', type=bool, default=False,
                    help='Use data streaming? (default: False)')
args = parser.parse_args()

test_size = 100
qtype_bit = hcl.UInt(1) # weights
qtype_int = hcl.Int(6) # not unsigned!
qtype_float = hcl.Fixed(20,10)
qtype_packed = hcl.UInt(32)

if __name__ == "__main__":
    target = None
    batch_size = 100
    dtype_in = qtype_bit
    dtype_out = qtype_float
else:
    batch_size = 1
    if args.vitis:
        print("[INFO] Use Vitis to compile")
        target = hcl.platform.aws_f1
        target.config(compile="vitis", mode="hw_exe")
        dtype_in = hcl.UInt(8)
        dtype_out = hcl.Fixed(32,10)
    else:
        target = hcl.platform.zc706
        target.config(compile="vivado_hls", mode="csyn")
        dtype_in = qtype_bit
        dtype_out = qtype_float

# compute declaration
def build_packed_bnn(input_image):
    w_conv1 = hcl.const_tensor(packed_params["w_conv1"],"w_conv1",qtype_bit)
    bn_t1 = hcl.const_tensor(packed_params["bn_t1"],"bn_t1",qtype_float)
    w_conv2 = hcl.const_tensor(packed_params["w_conv2"],"w_conv2",hcl.UInt(16))
    bn_t2 = hcl.const_tensor(packed_params["bn_t2"],"bn_t2",qtype_float)
    w_fc1 = hcl.const_tensor(packed_params["w_fc1"],"w_fc1",qtype_packed)
    b_fc1 = hcl.const_tensor(packed_params["b_fc1"],"b_fc1",qtype_float)
    w_fc2 = hcl.const_tensor(packed_params["w_fc2"],"w_fc2",qtype_packed)
    b_fc2 = hcl.const_tensor(packed_params["b_fc2"],"b_fc2",qtype_float)

    conv1 = bnn.packed_conv2d_nhwc(input_image, w_conv1, padding=[1,1], name="conv1", out_dtype=qtype_int)
    bn1 = bnn.packed_batch_norm_threshold_nhwc(conv1, bn_t1, name="bn1")
    if not args.stream:
        maxpool1 = bnn.packed_max_pool2d_nhwc(bn1, [2,2], [2,2], name="maxpool1")
    else:
        maxpool1 = bnn.packed_max_pool2d_nhwc_LB(bn1, [2,2], [2,2], name="maxpool1")

    conv2 = bnn.packed_conv2d_nhwc(maxpool1, w_conv2, padding=[1,1], name="conv2", out_dtype=qtype_int)
    bn2 = bnn.packed_batch_norm_threshold_nhwc(conv2, bn_t2, name="bn2")
    if not args.stream:
        maxpool2 = bnn.packed_max_pool2d_nhwc(bn2, [2,2], [2,2], name="maxpool2") # 32*4*4=512
    else:
        maxpool2 = bnn.packed_max_pool2d_nhwc_LB(bn2, [2,2], [2,2], name="maxpool2") # 32*4*4=512

    pack = bnn.packed_flatten_nhwc(maxpool2,name="packed_flatten")
    fc1 = bnn.packed_dense(pack, w_fc1, b_fc1, True, name="fc1") # 512/32->256/32
    fc2 = bnn.packed_dense(fc1, w_fc2, b_fc2, False, name="fc2") # 256/32->10
    return fc2

# prepare numpy arrays for testing
data = np.load("data/bnn-5775.data.npz")
images = data["images"][:test_size]
labels = data["labels"][:test_size]
num_images = images.shape[0]
params = np.load("data/bnn-5775.params.npz")

# prepare packed arrays
packed_params = {}
for name in params:
    if "w_fc" in name:
        packed_params[name] = np.packbits(params[name].copy().astype(np.bool),
            axis=1,bitorder="little").view(np.uint32)
    elif "w_conv1" in name:
        arr = params[name].copy().transpose(0,2,3,1).astype(np.bool)
        packed_params[name] = arr
    elif "w_conv2" in name:
        arr = params[name].copy().transpose(0,2,3,1)
        arr = np.packbits(arr.astype(np.bool),
                axis=3,bitorder="little").view(np.uint16)
        packed_params[name] = arr
    elif "bn_t" in name:
        packed_params[name] = params[name].copy().transpose(1,2,0)
    else:
        packed_params[name] = params[name].copy()

# declare hcl placeholders
def build_bitpacked_bnn_inf(batch_size=batch_size,target=target):
    print("build_bitpacked_bnn_inf")
    input_image = hcl.placeholder((batch_size,16,16,1),"input_image",dtype_in)
    s = hcl.create_schedule([input_image], build_packed_bnn)
    return hcl.build(s, target=target)

def build_bitpacked_bnn_inf_opt(batch_size=batch_size,target=target):
    print("build_bitpacked_bnn_inf_opt")
    # prepare placeholder
    input_image = hcl.placeholder((batch_size,16,16,1),"input_image",dtype_in)
    s = hcl.create_schedule([input_image], build_packed_bnn)

    # compute optimization
    layer_names = list(build_packed_bnn.__dict__.keys())
    new_layer = []
    for layer in layer_names:
        if not("w_" in layer or "bn_" in layer or "b_" in layer or "_LB" in layer):
            new_layer.append(layer)
    layer_names = new_layer
    s.partition(input_image,dim=3)
    for layer in layer_names:
        s_layer = getattr(build_packed_bnn,layer)
        if layer == "conv1_pad":
            s[s_layer].pipeline(s_layer.axis[1])
            if not args.stream:
                s.partition(s_layer,dim=3)
        elif layer == "conv1":
            s[s_layer].pipeline(s_layer.axis[2])
            LB = s.reuse_at(build_packed_bnn.conv1_pad._op,s[s_layer],s_layer.axis[1], "LB1")
            WB = s.reuse_at(LB,s[s_layer],s_layer.axis[2], "WB1")
            if not args.stream:
                s.partition(s_layer,dim=4)
        elif layer == "bn1":
            s[s_layer].pipeline(s_layer.axis[2])
            if not args.stream:
                s.partition(s_layer,dim=3)
        elif layer == "maxpool1":
            s[s_layer].pipeline(s_layer.axis[1])
            if not args.stream:
                s.partition(s_layer,dim=3)
        elif layer == "conv2_pad":
            s[s_layer].pipeline(s_layer.axis[1])
            if not args.stream:
                s.partition(s_layer,dim=3)
        elif layer == "conv2":
            s[s_layer].pipeline(s_layer.axis[2])
            LB = s.reuse_at(build_packed_bnn.conv2_pad._op,s[s_layer],s_layer.axis[1], "LB2")
            WB = s.reuse_at(LB,s[s_layer],s_layer.axis[2], "WB2")
            if not args.stream:
                s.partition(s_layer,dim=4)
        elif layer == "bn2":
            s[s_layer].pipeline(s_layer.axis[2])
            if not args.stream:
                s.partition(s_layer,dim=3)
        elif layer == "maxpool2":
            s[s_layer].pipeline(s_layer.axis[1])
            if not args.stream:
                s.partition(s_layer,dim=3)
        elif "unpack" in layer:
            s[s_layer].pipeline(s_layer.axis[1])
        elif layer == "packed_flatten":
            s[s_layer].pipeline(s_layer.axis[1])
        elif layer == "fc1_matmul":
            s[s_layer].pipeline(s_layer.axis[2])
            s_fc1 = build_packed_bnn.fc1
            s[s_fc1].pipeline(s_fc1.axis[2])
        elif layer == "fc2_matmul":
            s[s_layer].pipeline(s_layer.axis[2])
            s_fc2 = build_packed_bnn.fc2
            s[s_fc2].pipeline(s_fc2.axis[1])

    # streaming across layers
    if args.stream:
        print("[INFO] Use stream")
        for i,layer in enumerate(layer_names):
            if i == len(layer_names) - 1:
                break
            layer1 = getattr(build_packed_bnn,layer)
            layer2 = getattr(build_packed_bnn,list(layer_names)[i+1])
            s.to(layer1,s[layer2])

    return hcl.build(s, target=target)

if __name__ == '__main__':

    print("build_bitpacked_bnn_inf")
    f = build_bitpacked_bnn_inf()
    print("Finish building function.")

    correct_sum = 0
    for i in range(num_images // batch_size):
        np_image = images[i*batch_size:(i+1)*batch_size].transpose(0,2,3,1)
        hcl_image = hcl.asarray(np_image, dtype=dtype_in)
        hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=dtype_out)
        f(hcl_image, hcl_out)
        prediction = np.argmax(hcl_out.asnumpy(), axis=1)
        correct_sum += np.sum(np.equal(prediction, labels[i*batch_size:(i+1)*batch_size]))
        if (i+1) % 10 == 0:
            print("Done {} batches.".format(i+1))
    print("Testing accuracy: {}".format(correct_sum / float(num_images)))
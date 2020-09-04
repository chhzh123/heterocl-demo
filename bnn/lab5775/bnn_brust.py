import heterocl as hcl
import hlib.op.bnn as bnn
import numpy as np
import sys
from heterocl.profiler import Profiler

profiler = Profiler()

target = None
test_size = 100
batch_size = 100
qtype_bit = hcl.UInt(1) # weights
qtype_int = hcl.Int(6) # not unsigned!
qtype_float = hcl.Fixed(20,10)
qtype_packed = hcl.UInt(32)

def build_packed_bnn(*arrays): # 1*16*16
    hcl_comp = []
    for i,array in enumerate(arrays):
        if i in [0,1]:
            dtype = qtype_bit
        elif i == 3:
            dtype = hcl.UInt(16)
        elif i in [5,7]:
            dtype = qtype_packed
        else:
            dtype = qtype_float
        hcl_comp.append(hcl.compute(array.shape,lambda *dim: array[dim],name="copy_{}".format(i),dtype=dtype))
    input_image = hcl_comp[0]
    w_conv1 = hcl_comp[1]
    bn_t1 = hcl_comp[2]
    w_conv2 = hcl_comp[3]
    bn_t2 = hcl_comp[4]
    w_fc1 = hcl_comp[5]
    b_fc1 = hcl_comp[6]
    w_fc2 = hcl_comp[7]
    b_fc2 = hcl_comp[8]

    conv1 = bnn.packed_conv2d_nchw(input_image, w_conv1, padding=[1,1], name="conv1", out_dtype=qtype_int) # 16*16*16
    bn1 = bnn.packed_batch_norm_threshold(conv1, bn_t1, name="bn1")
    # maxpool1 = bnn.packed_max_pool2d_nchw(bn1, [2,2], [2,2], name="maxpool1",unpack=False) # 16*8*8
    maxpool1 = bnn.packed_max_pool2d_LB(bn1, [2,2], [2,2], name="maxpool1")

    conv2 = bnn.packed_conv2d_nchw(maxpool1, w_conv2, padding=[1,1], name="conv2", out_dtype=qtype_int) # 32*8*8
    bn2 = bnn.packed_batch_norm_threshold(conv2, bn_t2, name="bn2")
    # maxpool2 = bnn.packed_max_pool2d_nchw(bn2, [2,2], [2,2], name="maxpool2",unpack=False) # 16*8*8
    maxpool2 = bnn.packed_max_pool2d_LB(bn2, [2,2], [2,2], name="maxpool2") # 32*4*4=512

    pack = bnn.packed_flatten(maxpool2,name="packed_flatten")
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
    elif "w_conv2" in name:
        arr = params[name].copy().transpose(0,2,3,1)
        arr = np.packbits(arr.astype(np.bool),
                axis=3,bitorder="little").view(np.uint16)
        packed_params[name] = arr.transpose(0,3,1,2)
    else:
        packed_params[name] = params[name].copy()

def build_bitpacked_bnn_inf(batch_size=batch_size,target=target):
    # prepare placeholder
    hcl_ph = []
    input_image = hcl.placeholder((batch_size,1,16,16),"input_image",qtype_bit)
    for name in packed_params:
        if "w_conv2" in name:
            dtype = hcl.UInt(16)
        else:
            dtype = qtype_bit if "conv" in name else (qtype_packed if "w_fc" in name else qtype_float)
        hcl_ph.append(hcl.placeholder(packed_params[name].shape,name,dtype=dtype))

    # build the network
    s = hcl.create_schedule([input_image] + hcl_ph, build_packed_bnn)
    layer_names = build_packed_bnn.__dict__.keys()

    if isinstance(target,hcl.platform):
        s.to([input_image] + hcl_ph, target.xcel)
        s.to(build_packed_bnn.fc2, target.host)

    return hcl.build(s, target=target)

def build_bitpacked_bnn_inf_opt(batch_size=batch_size,target=target):
    # prepare placeholder
    hcl_ph = []
    ph_dict = {}
    input_image = hcl.placeholder((batch_size,1,16,16),"input_image",qtype_bit)
    for name in packed_params:
        if "w_conv2" in name:
            dtype = hcl.UInt(16)
        else:
            dtype = qtype_bit if "conv" in name else (qtype_packed if "w_fc" in name else qtype_float)
        hcl_ph.append(hcl.placeholder(packed_params[name].shape,name,dtype=dtype))
        ph_dict[name] = hcl_ph[-1]

    # build the network
    s = hcl.create_schedule([input_image] + hcl_ph, build_packed_bnn)

    # compute optimization
    layer_names = build_packed_bnn.__dict__.keys()
    for layer in layer_names:
        s_layer = getattr(build_packed_bnn,layer)
        if layer == "conv1_pad":
            s[s_layer].pipeline(s_layer.axis[2])
            s.partition(input_image)
            s.partition(s_layer,dim=4)
        elif layer == "conv2_pad":
            s[s_layer].pipeline(s_layer.axis[2])
            s.partition(s_layer,dim=4)
        elif layer == "bn1":
            s_conv = build_packed_bnn.conv1
            s[s_conv].pipeline(s_conv.axis[3])
            s[s_layer].pipeline(s_layer.axis[3])
            LB = s.reuse_at(build_packed_bnn.conv1_pad._op,s[s_conv],s_conv.axis[2], "LB1")
            WB = s.reuse_at(LB,s[s_conv],s_conv.axis[3], "WB1")
            s.partition(LB, dim=3)
            s.partition(WB)
            s.partition(build_packed_bnn.copy_1)
            s.partition(build_packed_bnn.copy_2,dim=1)
            s.partition(build_packed_bnn.conv1,dim=2)
            s.partition(s_layer,dim=4)
        elif layer == "maxpool1":
            s.partition(s_layer,dim=4)
            s[s_layer].pipeline(s_layer.axis[2])
        elif layer == "bn2":
            s_conv = build_packed_bnn.conv2
            s[s_layer].pipeline(s_layer.axis[3]) # be careful of # channels
            s[s_conv].pipeline(s_conv.axis[3])
            LB = s.reuse_at(build_packed_bnn.conv2_pad._op,s[s_conv],s_conv.axis[2], "LB2")
            WB = s.reuse_at(LB,s[s_conv],s_conv.axis[3], "WB2")
            s.partition(LB, dim=3)
            s.partition(WB)
            s.partition(build_packed_bnn.copy_3)
            s.partition(build_packed_bnn.copy_4,dim=1)
            s.partition(build_packed_bnn.conv2,dim=2)
            s.partition(s_layer,dim=4)
        elif layer == "maxpool2":
            s.partition(s_layer,dim=4)
            s[s_layer].pipeline(s_layer.axis[2])
        elif "unpack" in layer:
            s[s_layer].pipeline(s_layer.axis[1])
        elif layer == "packed_flatten":
            s[s_layer].pipeline(s_layer.axis[1])
        elif layer == "fc1_matmul":
            s[s_layer].pipeline(s_layer.axis[2])
            s_fc1 = build_packed_bnn.fc1
            s[s_fc1].pipeline(s_fc1.axis[1])
        elif layer == "fc2_matmul":
            s[s_layer].pipeline(s_layer.axis[2])
            s_fc2 = build_packed_bnn.fc2
            s[s_fc2].pipeline(s_fc2.axis[1])

    # streaming across layers
    new_layer_names = []
    for layer in list(layer_names):
        if not ("LB" in layer or "copy" in layer):
            new_layer_names.append(layer)
        elif "copy" in layer:
            s_layer = getattr(build_packed_bnn,layer)
            s[s_layer].pipeline(s_layer.axis[len(s_layer.axis)-1])
    layer_names = new_layer_names
    for i,layer in enumerate(layer_names):
        if i == len(layer_names) - 1:
            break
        layer1 = getattr(build_packed_bnn,layer)
        layer2 = getattr(build_packed_bnn,list(layer_names)[i+1])
        s.to(layer1,s[layer2])

    if isinstance(target,hcl.platform):
        s.to([input_image] + hcl_ph, target.xcel)
        s.to(build_packed_bnn.fc2, target.host)

    return hcl.build(s, target=target)

if __name__ == '__main__':

    hcl_array = []
    for name in packed_params:
        if "w_conv2" in name:
            dtype = hcl.UInt(16)
        else:
            dtype = qtype_bit if "conv" in name else (qtype_packed if "w_fc" in name else qtype_float)
        hcl_array.append(hcl.asarray(packed_params[name],dtype=dtype))
    hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)
    f = build_bitpacked_bnn_inf()
    print("Finish building function.")

    correct_sum = 0
    for i in range(num_images // batch_size):
        np_image = images[i*batch_size:(i+1)*batch_size]
        hcl_image = hcl.asarray(np_image, dtype=qtype_bit)
        f(hcl_image, *hcl_array, hcl_out)
        prediction = np.argmax(hcl_out.asnumpy(), axis=1)
        correct_sum += np.sum(np.equal(prediction, labels[i*batch_size:(i+1)*batch_size]))
        if (i+1) % 10 == 0:
            print("Done {} batches.".format(i+1))
    print("Testing accuracy: {}".format(correct_sum / float(num_images)))
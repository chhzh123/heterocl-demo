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

def build_packed_bnn(input_image, w_conv1, bn_t1,
                     w_conv2, bn_t2,
                     w_fc1, b_fc1,
                     w_fc2, b_fc2): # 16*16*1
    conv1 = bnn.packed_conv2d_nhwc(input_image, w_conv1, padding=[1,1], name="conv1", out_dtype=qtype_int)
    bn1 = bnn.packed_batch_norm_threshold_nhwc(conv1, bn_t1, name="bn1")
    maxpool1 = bnn.packed_max_pool2d_nhwc(bn1, [2,2], [2,2], name="maxpool1")

    conv2 = bnn.packed_conv2d_nhwc(maxpool1, w_conv2, padding=[1,1], name="conv2", out_dtype=qtype_int)
    bn2 = bnn.packed_batch_norm_threshold_nhwc(conv2, bn_t2, name="bn2")
    maxpool2 = bnn.packed_max_pool2d_nhwc(bn2, [2,2], [2,2], name="maxpool2") # 32*4*4=512

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

def build_bitpacked_bnn_inf(batch_size=batch_size,target=target):
    # prepare placeholder
    hcl_ph = []
    input_image = hcl.placeholder((batch_size,16,16,1),"input_image",qtype_bit)
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
        np_image = images[i*batch_size:(i+1)*batch_size].transpose(0,2,3,1)
        hcl_image = hcl.asarray(np_image, dtype=qtype_bit)
        f(hcl_image, *hcl_array, hcl_out)
        prediction = np.argmax(hcl_out.asnumpy(), axis=1)
        correct_sum += np.sum(np.equal(prediction, labels[i*batch_size:(i+1)*batch_size]))
        if (i+1) % 10 == 0:
            print("Done {} batches.".format(i+1))
    print("Testing accuracy: {}".format(correct_sum / float(num_images)))
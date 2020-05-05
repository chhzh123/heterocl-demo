import heterocl as hcl
import hlib
import numpy as np

target = None
test_size = 100
batch_size = 100
qtype_bit = hcl.UInt(1) # weights
qtype_int = hcl.Int(10) # not unsigned!
qtype_float = hcl.Float()

# compute declaration
def build_bnn(input_image, w_conv1, bn_t1,
              w_conv2, bn_t2,
              w_fc1, b_fc1,
              w_fc2, b_fc2): # 1*16*16
    conv1 = hlib.op.bnn.conv2d_nchw(input_image, w_conv1, padding=[1,1], name="conv1") # 16*16*16
    bn1 = hlib.op.bnn.batch_norm_threshold(conv1, bn_t1)
    maxpool1 = hlib.op.bnn.max_pool2d_nchw(bn1, [2,2], [2,2]) # 16*8*8
    conv2 = hlib.op.bnn.conv2d_nchw(maxpool1, w_conv2, padding=[1,1], name="conv2") # 32*8*8
    bn2 = hlib.op.bnn.batch_norm_threshold(conv2, bn_t2)
    maxpool2 = hlib.op.bnn.max_pool2d_nchw(bn2, [2,2], [2,2]) # 32*4*4=512
    flat = hlib.op.bnn.flatten(maxpool2)
    fc1 = hlib.op.bnn.dense(flat, w_fc1, b_fc1, True) # 512->256
    fc2 = hlib.op.bnn.dense(fc1, w_fc2, b_fc2, False) # 256->10
    return fc2

# prepare numpy arrays for testing
data = np.load("data/bnn-5775.data.npz")
images = data["images"][:test_size]
labels = data["labels"][:test_size]
num_images = images.shape[0]
params = np.load("data/bnn-5775.params.npz")

# declare hcl placeholders
def build_bnn_inf(batch_size=batch_size,target=target):
    hcl_ph = []
    input_image = hcl.placeholder((batch_size,1,16,16),"input_image",qtype_int)
    for name in params:
        dtype = qtype_bit if ("conv" in name or "w_" in name) else qtype_float
        hcl_ph.append(hcl.placeholder(params[name].shape,name,dtype=dtype))

    # build the network
    scheme = hcl.create_scheme([input_image] + hcl_ph, build_bnn)
    s = hcl.create_schedule_from_scheme(scheme)
    return hcl.build(s, target=target)

if __name__ == '__main__':

    f = build_bnn_inf()

    hcl_array = []
    for name in params:
        dtype = qtype_bit if ("conv" in name or "w_" in name) else qtype_float
        hcl_array.append(hcl.asarray(params[name],dtype=dtype))
    hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)

    correct_sum = 0
    for i in range(num_images // batch_size):
        np_image = images[i*batch_size:(i+1)*batch_size]
        hcl_image = hcl.asarray(np_image, dtype=qtype_int)
        f(hcl_image, *hcl_array, hcl_out)
        prediction = np.argmax(hcl_out.asnumpy(), axis=1)
        correct_sum += np.sum(np.equal(prediction, labels[i*batch_size:(i+1)*batch_size]))
        if (i+1) % 10 == 0:
            print("Done {} batches.".format(i+1))
    print("Testing accuracy: {}".format(correct_sum / float(num_images)))
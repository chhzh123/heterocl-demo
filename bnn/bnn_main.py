import heterocl as hcl
import hlib
import numpy as np

hcl.init(hcl.Float())

# compute declaration
def build_bnn(input_image, w_conv1,
              gamma1, beta1, miu1, sigma1,
              w_conv2,
              gamma2, beta2, miu2, sigma2,
              w_fc1, b_fc1,
              w_fc2, b_fc2): # 1*16*16
    conv1 = hlib.op.bnn.conv2d_nchw(input_image, w_conv1, padding=[1,1], name="conv1") # 64*16*16
    bn1 = hlib.op.bnn.batch_norm(conv1, gamma1, beta1, miu1, sigma1)[0]
    maxpool1 = hlib.op.bnn.max_pool2d_nchw(bn1, [2,2], [2,2]) # 64*8*8
    conv2 = hlib.op.bnn.conv2d_nchw(maxpool1, w_conv2, padding=[1,1], name="conv2") # 128*8*8
    bn2 = hlib.op.bnn.batch_norm(conv2, gamma2, beta2, miu2, sigma2, 64)[0]
    maxpool2 = hlib.op.bnn.max_pool2d_nchw(bn2, [2,2], [2,2]) # 128*4*4=2048
    flat = hlib.op.bnn.flatten(maxpool2)
    fc1 = hlib.op.bnn.dense(flat, w_fc1, b_fc1, True) # 2048->512
    fc2 = hlib.op.bnn.dense(fc1, w_fc2, b_fc2, False) # 512->10
    return fc2

target = None
batch_size = 100
qtype_bit = hcl.Int()
qtype_float = hcl.Float()

# prepare the numpy arrays for testing
images = np.loadtxt("data/test_b.dat").astype(np.int).reshape((-1,1,16,16)) # 5000 images
labels = np.loadtxt("data/label.dat").astype(np.int)
num_images = images.shape[0]
print("Loaded {} images".format(num_images))
params = {
    "w_conv1": np.loadtxt("data/weight_0b",delimiter=",").astype(np.int).reshape((1,64,3,3)).transpose(1,0,2,3), # 576
    "gamma1" : np.loadtxt("data/weight_1p",delimiter=",").astype(np.float),
    "beta1"  : np.loadtxt("data/weight_2p",delimiter=",").astype(np.float),
    "miu1"   : np.loadtxt("data/weight_3p",delimiter=",").astype(np.float),
    "sigma1" : np.loadtxt("data/weight_4p",delimiter=",").astype(np.float),
    "w_conv2": np.loadtxt("data/weight_5b",delimiter=",").astype(np.int).reshape((64,128,3,3)).transpose(1,0,2,3),
    "gamma2" : np.loadtxt("data/weight_6p",delimiter=",").astype(np.float),
    "beta2"  : np.loadtxt("data/weight_7p",delimiter=",").astype(np.float),
    "miu2"   : np.loadtxt("data/weight_8p",delimiter=",").astype(np.float),
    "sigma2" : np.loadtxt("data/weight_9p",delimiter=",").astype(np.float),
    "w_fc1"  : np.loadtxt("data/weight_10b",delimiter=",").astype(np.float).reshape((2048,512)).T,
    "b_fc1"  : np.loadtxt("data/weight_11p",delimiter=",").astype(np.float),
    "w_fc2"  : np.loadtxt("data/weight_12b",delimiter=",").astype(np.float).reshape((512,10)).T,
    "b_fc2"  : np.loadtxt("data/weight_13p",delimiter=",").astype(np.float)
}

# declare hcl array and placeholders
hcl_array = []
hcl_ph = []
input_image = hcl.placeholder((batch_size,1,16,16),"input_image",qtype_bit)
for name in params:
    dtype = qtype_bit if "conv" in name else qtype_float
    hcl_array.append(hcl.asarray(params[name],dtype=dtype))
    hcl_ph.append(hcl.placeholder(params[name].shape,name,dtype=dtype))
hcl_out = hcl.asarray(np.zeros((batch_size,10)))

scheme = hcl.create_scheme([input_image] + hcl_ph, build_bnn)
s = hcl.create_schedule_from_scheme(scheme)
f = hcl.build(s, target=target)

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
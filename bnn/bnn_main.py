import heterocl as hcl
import hlib
import numpy as np

hcl.init(hcl.Float())

def build_bnn(input_image, w_conv1, w_conv2,
              gamma1, beta1, miu1, sigma1,
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

batch_size = 1
qtype_bit = hcl.Int()
qtype_float = hcl.Float()

def build_bnn_inf(batch_size=batch_size, target=None):
    # set up input/output placeholders
    input_image = hcl.placeholder((batch_size,1,16,16),"input_image",qtype_bit)

    w_conv1 = hcl.placeholder((64,1,3,3),"conv1",qtype_bit)
    gamma1 = hcl.placeholder((64,),"gamma1",qtype_float) # (channels,)
    beta1 = hcl.placeholder((64,),"beta1",qtype_float)
    miu1 = hcl.placeholder((64,),"miu1",qtype_float)
    sigma1 = hcl.placeholder((64,),"sigma1",qtype_float)

    w_conv2 = hcl.placeholder((128,64,3,3), "conv2",qtype_bit)
    gamma2 = hcl.placeholder((128,),"gamma2",qtype_float)
    beta2 = hcl.placeholder((128,),"beta2",qtype_float)
    miu2 = hcl.placeholder((128,),"miu2",qtype_float)
    sigma2 = hcl.placeholder((128,),"sigma2",qtype_float)

    w_fc1 = hcl.placeholder((512,2048),"w_fc1",qtype_float)
    b_fc1 = hcl.placeholder((512,),"b_fc1",qtype_float)
    w_fc2 = hcl.placeholder((10,512),"w_fc2",qtype_float)
    b_fc2 = hcl.placeholder((10,),"b_fc2",qtype_float)

    # create a quantization scheme
    scheme = hcl.create_scheme([input_image, w_conv1, w_conv2,
                                gamma1, beta1, miu1, sigma1,
                                gamma2, beta2, miu2, sigma2,
                                w_fc1, b_fc1,
                                w_fc2, b_fc2], build_bnn)
    # quantize the layers
    # scheme.quantize([build_bnn.conv1, build_bnn.conv2], qtype_bit)
    s = hcl.create_schedule_from_scheme(scheme)
    return hcl.build(s, target=target)

f = build_bnn_inf()

# prepare the numpy arrays for testing
images = np.loadtxt("data/test_b.dat").astype(np.int).reshape((-1,1,16,16)) # 5000 images
print("Loaded {} images".format(images.shape[0]))
np_w_conv1 = np.loadtxt("data/weight_0b",delimiter=",").astype(np.int).reshape((1,64,3,3)) # 576
np_w_conv1 = np_w_conv1.transpose(1,0,2,3)
np_gamma1 = np.loadtxt("data/weight_1p",delimiter=",").astype(np.float)
np_beta1 = np.loadtxt("data/weight_2p",delimiter=",").astype(np.float)
np_miu1 = np.loadtxt("data/weight_3p",delimiter=",").astype(np.float)
np_sigma1 = np.loadtxt("data/weight_4p",delimiter=",").astype(np.float)
np_w_conv2 = np.loadtxt("data/weight_5b",delimiter=",").astype(np.int).reshape((64,128,3,3))
np_w_conv2 = np_w_conv2.transpose(1,0,2,3)
np_gamma2 = np.loadtxt("data/weight_6p",delimiter=",").astype(np.float)
np_beta2 = np.loadtxt("data/weight_7p",delimiter=",").astype(np.float)
np_miu2 = np.loadtxt("data/weight_8p",delimiter=",").astype(np.float)
np_sigma2 = np.loadtxt("data/weight_9p",delimiter=",").astype(np.float)
np_w_fc1 = np.loadtxt("data/weight_10b",delimiter=",").astype(np.float).reshape((2048,512))
np_w_fc1 = np_w_fc1.T
np_b_fc1 = np.loadtxt("data/weight_11p",delimiter=",").astype(np.float)
np_w_fc2 = np.loadtxt("data/weight_12b",delimiter=",").astype(np.float).reshape((512,10))
np_w_fc2 = np_w_fc2.T
np_b_fc2 = np.loadtxt("data/weight_13p",delimiter=",").astype(np.float)
np_image = images[:1,:,:,:] # batch_size=1

hcl_image = hcl.asarray(np_image, dtype=qtype_bit)

hcl_w_conv1 = hcl.asarray(np_w_conv1, dtype=qtype_bit)
hcl_gamma1 = hcl.asarray(np_gamma1, dtype=qtype_float)
hcl_beta1 = hcl.asarray(np_beta1, dtype=qtype_float)
hcl_miu1 = hcl.asarray(np_miu1, dtype=qtype_float)
hcl_sigma1 = hcl.asarray(np_sigma1, dtype=qtype_float)

hcl_w_conv2 = hcl.asarray(np_w_conv2, dtype=qtype_bit)
hcl_gamma2 = hcl.asarray(np_gamma2, dtype=qtype_float)
hcl_beta2 = hcl.asarray(np_beta2, dtype=qtype_float)
hcl_miu2 = hcl.asarray(np_miu2, dtype=qtype_float)
hcl_sigma2 = hcl.asarray(np_sigma2, dtype=qtype_float)

hcl_w_fc1 = hcl.asarray(np_w_fc1, dtype=qtype_float)
hcl_b_fc1 = hcl.asarray(np_b_fc1, dtype=qtype_float)
hcl_w_fc2 = hcl.asarray(np_w_fc2, dtype=qtype_float)
hcl_b_fc2 = hcl.asarray(np_b_fc2, dtype=qtype_float)
hcl_out = hcl.asarray(np.zeros((1,10)))
f(hcl_image, hcl_w_conv1, hcl_w_conv2,
    hcl_gamma1, hcl_beta1, hcl_miu1, hcl_sigma1,
    hcl_gamma2, hcl_beta2, hcl_miu2, hcl_sigma2,
    hcl_w_fc1, hcl_b_fc1,
    hcl_w_fc2, hcl_b_fc2,
    hcl_out)
print(hcl_out)
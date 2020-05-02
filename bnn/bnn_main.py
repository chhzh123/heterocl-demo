import heterocl as hcl
import hlib
import numpy as np

hcl.init(hcl.Int())

def build_bnn(input_image, weight_conv1):
    # first conv
    # 1*16*16
    return hlib.op.nn.conv2d_nchw(input_image, weight_conv1)#, padding=[2,2]) # 64*16*16
    # return conv1
    # tanh1 = hlib.bnn.tanh(conv1, "tanh1")
    # pool1 = hlib.bnn.max_pool(tanh1, kernel=(2,2), stride=(2,2)) # 32*14*14
    # # second conv
    # conv2 = hlib.bnn.conv2d_nchw(pool1, weight_conv2) # 64*14*14
    # tanh2 = hlib.bnn.tanh(conv2, "tanh2")
    # pool2 = hlib.bnn.max_pool(tanh2, kernel=(2,2), stride=(2,2)) # 64*7*7
    # # first fc
    # flat = hlib.bnn.flatten(pool2) # 3136
    # fc1 = hlib.bnn.dense(flat, weight_fc1) # 512
    # # second fc
    # batch_norm
    # fc2 =  hlib.bnn.dense(tanh3, weight_fc2) # 10
    # # loss
    # return fc2

A = hcl.placeholder((1,1,16,16), "A")
weight_conv = hcl.placeholder((64,1,3,3), "conv")
s = hcl.create_scheme([A, weight_conv], build_bnn)
# s.downsize(, hcl.UInt(1))
s = hcl.create_schedule_from_scheme(s)
f = hcl.build(s, target=None)
# print(f)

images = np.loadtxt("data/test_b.dat").astype(np.int).reshape((-1,1,16,16)) # 5000 images
print("Loaded {} images".format(images.shape[0]))
np_weight = np.loadtxt("data/weight_0b",delimiter=",").astype(np.int).reshape((64,1,3,3)) # 576
np_A = images[:1,:,:,:]
hcl_A = hcl.asarray(np_A)
hcl_weight = hcl.asarray(np_weight)
hcl_B = hcl.asarray(np.zeros((1,64,16,16)))
f(hcl_A,hcl_weight,hcl_B)
print(hcl_B)
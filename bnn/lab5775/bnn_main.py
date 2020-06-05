import heterocl as hcl
import hlib.op.bnn as bnn
import numpy as np
import sys

target = None
test_size = 100
batch_size = 100
qtype_bit = hcl.UInt(1) # weights
qtype_int = hcl.Int(6) # not unsigned!
qtype_float = hcl.Fixed(20,10)
qtype_packed = hcl.UInt(32)

# compute declaration
def build_bnn(input_image, w_conv1, bn_t1,
              w_conv2, bn_t2,
              w_fc1, b_fc1,
              w_fc2, b_fc2): # 1*16*16
    conv1 = bnn.conv2d_nchw(input_image, w_conv1, padding=[1,1], name="conv1",out_dtype=qtype_int) # 16*16*16
    bn1 = bnn.batch_norm_threshold(conv1, bn_t1, name="bn1")
    maxpool1 = bnn.max_pool2d_nchw(bn1, [2,2], [2,2], name="maxpool1") # 16*8*8
    conv2 = bnn.conv2d_nchw(maxpool1, w_conv2, padding=[1,1], name="conv2",out_dtype=qtype_int) # 32*8*8
    bn2 = bnn.batch_norm_threshold(conv2, bn_t2, name="bn2")
    maxpool2 = bnn.max_pool2d_nchw(bn2, [2,2], [2,2], name="maxpool2") # 32*4*4=512
    flat = bnn.flatten(maxpool2, name="flatten")
    fc1 = bnn.dense(flat, w_fc1, b_fc1, True, name="fc1") # 512->256
    fc2 = bnn.dense(fc1, w_fc2, b_fc2, False, name="fc2") # 256->10
    return fc2

counts = hcl.array(np.array(list(bytes(bin(i).count("1") for i in range(256)))))
PACK_CONV = True

def build_packed_bnn(input_image, w_conv1, bn_t1,
                     w_conv2, bn_t2,
                     w_fc1, b_fc1,
                     w_fc2, b_fc2): # 1*16*16
    if PACK_CONV:
        conv1 = bnn.packed_conv2d_nchw(input_image, w_conv1, padding=[1,1], name="conv1",out_dtype=qtype_int) # 16*16*16
    else:
        conv1 = bnn.conv2d_nchw(input_image, w_conv1, padding=[1,1], name="conv1",out_dtype=qtype_int) # 16*16*16
    bn1 = bnn.batch_norm_threshold(conv1, bn_t1, name="bn1")
    maxpool1 = bnn.packed_max_pool2d_nchw(bn1, [2,2], [2,2], name="maxpool1",unpack=True) # 16*8*8

    if False:
        w = hcl.pack(w_conv2, axis=1, factor=16, dtype=hcl.UInt(16), name="w")
        conv2 = bnn.packed_conv2d_nchw(maxpool1, w, padding=[1,1], name="conv2",out_dtype=qtype_int) # 32*8*8
    else:
        conv2 = bnn.conv2d_nchw(maxpool1, w_conv2, padding=[1,1], name="conv2",out_dtype=qtype_int) # 32*8*8
    bn2 = bnn.batch_norm_threshold(conv2, bn_t2, name="bn2")
    maxpool2 = bnn.packed_max_pool2d_nchw(bn2, [2,2], [2,2], name="maxpool2",unpack=False) # 32*4*4=512

    if PACK_CONV:
        pack = bnn.packed_flatten(maxpool2,name="packed_flatten")
    else:
        flat = bnn.flatten(maxpool2, name="flatten")
        pack = hcl.pack(flat, axis=1, factor=32, dtype=qtype_packed, name="pack") # 512/32=16
    fc1 = bnn.packed_dense(pack, w_fc1, b_fc1, True, name="fc1") # 512/32->256/32
    fc2 = bnn.packed_dense(fc1, w_fc2, b_fc2, False, name="fc2") # 256/32->10
    return fc2

def packbits(arr, bitwidth):
    assert arr.dtype == np.bool, \
        "A bool array is needed"
    return np.packbits(arr,axis=1,bitorder="little").view(np.uint32)

# prepare numpy arrays for testing
data = np.load("data/bnn-5775.data.npz")
images = data["images"][:test_size]
labels = data["labels"][:test_size]
num_images = images.shape[0]
params = np.load("data/bnn-5775.params.npz")

# prepare packed arrays
# packed_images = packbits(images.astype(np.bool),32)
# packed_labels = packbits(labels.astype(np.bool),32)
packed_params = {}
for name in params:
    if "w_fc" in name:
        packed_params[name] = packbits(params[name].copy().astype(np.bool),32)
    # elif "w_conv2" in name:
        # packed_params[name] = np.packbits(arr,axis=1,bitorder="little").view(np.uint16)
    else:
        packed_params[name] = params[name].copy()

# declare hcl placeholders
def build_bnn_inf(batch_size=batch_size,target=target):
    hcl_ph = []
    input_image = hcl.placeholder((batch_size,1,16,16),"input_image",qtype_bit)
    for name in params:
        dtype = qtype_bit if ("conv" in name or "w_" in name) else qtype_float
        hcl_ph.append(hcl.placeholder(params[name].shape,name,dtype=dtype))

    # build the network
    scheme = hcl.create_scheme([input_image] + hcl_ph, build_bnn)
    s = hcl.create_schedule_from_scheme(scheme)

    if isinstance(target,hcl.platform):
        s.to([input_image] + hcl_ph, target.xcel)
        s.to(build_bnn.fc2, target.host)
        target.config(compile="vivado_hls", mode="csyn")

    return hcl.build(s, target=target)

def build_bnn_inf_opt(batch_size=batch_size,target=target):
    hcl_ph = []
    input_image = hcl.placeholder((batch_size,1,16,16),"input_image",qtype_bit)
    for name in params:
        dtype = qtype_bit if ("conv" in name or "w_" in name) else qtype_float
        hcl_ph.append(hcl.placeholder(params[name].shape,name,dtype=dtype))

    # build the network
    scheme = hcl.create_scheme([input_image] + hcl_ph, build_bnn)
    s = hcl.create_schedule_from_scheme(scheme)

    def plot_dataflow_graph():
        import matplotlib.pyplot as plt
        import networkx as nx
        graph, op = s.dataflow_graph(plot=True)
        nx.draw(graph, with_labels=True)
        plt.savefig("bnn.png")

    # compute optimization
    for layer in build_bnn.__dict__.keys():
        s_layer = getattr(build_bnn,layer)
        if "bn" in layer: # fuse conv
            s_conv = getattr(build_bnn,"conv" + layer[-1])
            s[s_conv].compute_at(s[s_layer],s_layer.axis[3])
            if layer == "bn1":
                s[s_layer].pipeline(s_layer.axis[3]) # will be refreshed
            else:
                s[s_conv].pipeline(s_conv.axis[4])
        elif "pool" in layer:
            s[s_layer].pipeline(s_layer.axis[2])
        elif "fc" in layer:
            s[s_layer].pipeline(s_layer.axis[1])
        elif "flatten" in layer:
            s[s_layer].pipeline(s_layer.axis[1])
        elif "dense_relu" in layer:
            s_fc = getattr(build_bnn,"fc1")
            s[s_fc].compute_at(s[s_layer],s_layer.axis[1])
            s[s_fc].pipeline(s_fc.axis[2])

    if isinstance(target,hcl.platform):
        s.to([input_image] + hcl_ph, target.xcel)
        s.to(build_bnn.fc2, target.host)
        target.config(compile="vivado_hls", mode="csyn")

    # memory optimization
    s.partition(input_image, hcl.Partition.Block, dim=1, factor=8)
    for ph in reversed(hcl_ph):
        if ph.name in ["b_fc2", "fc2"]:
            s.partition(ph, hcl.Partition.Complete, dim=1)
        else:
            s.partition(ph, hcl.Partition.Block, dim=1, factor=8)

    return hcl.build(s, target=target)

def build_bitpacked_bnn_inf(batch_size=batch_size,target=target):
    # prepare placeholder
    hcl_ph = []
    input_image = hcl.placeholder((batch_size,1,16,16),"input_image",qtype_bit)
    for name in packed_params:
        dtype = qtype_bit if "conv" in name else (qtype_packed if "w_fc" in name else qtype_float)
        hcl_ph.append(hcl.placeholder(packed_params[name].shape,name,dtype=dtype))

    # build the network
    scheme = hcl.create_scheme([input_image] + hcl_ph, build_packed_bnn)
    s = hcl.create_schedule_from_scheme(scheme)

    if isinstance(target,hcl.platform):
        s.to([input_image] + hcl_ph, target.xcel)
        s.to(build_packed_bnn.fc2, target.host)
        target.config(compile="vivado_hls", mode="csyn")

    return hcl.build(s, target=target)

def build_bitpacked_bnn_inf_opt(batch_size=batch_size,target=target):
    # prepare placeholder
    hcl_ph = []
    input_image = hcl.placeholder((batch_size,1,16,16),"input_image",qtype_bit)
    for name in packed_params:
        dtype = qtype_bit if "conv" in name else (qtype_packed if "w_fc" in name else qtype_float)
        hcl_ph.append(hcl.placeholder(packed_params[name].shape,name,dtype=dtype))

    # build the network
    scheme = hcl.create_scheme([input_image] + hcl_ph, build_packed_bnn)
    s = hcl.create_schedule_from_scheme(scheme)

    # compute optimization
    print(build_packed_bnn.__dict__.keys())
    for layer in build_packed_bnn.__dict__.keys():
        s_layer = getattr(build_packed_bnn,layer)
        if "pad" in layer:
            s[s_layer].pipeline(s_layer.axis[1])
        elif layer == "bn1": # fuse conv
            s_conv = build_packed_bnn.conv1
            s[s_conv].compute_at(s[s_layer],s_layer.axis[3])
            s[s_layer].pipeline(s_layer.axis[2])
            # s_packed = build_packed_bnn.packed_bn1
            # s[s_layer].compute_at(s[s_packed],s_packed.axis[3])
        # elif layer in ["pack2"]:
        #     s[s_layer].pipeline(s_layer.axis[1])
        elif layer == "maxpool1":
            s[s_layer].pipeline(s_layer.axis[2])
        elif layer == "bn2":
            s_conv = build_packed_bnn.conv2
            s[s_layer].pipeline(s_layer.axis[3]) # be careful of # channels
            s[s_conv].pipeline(s_conv.axis[3])
            LB = s.reuse_at(build_packed_bnn.conv2_pad._op,s[s_conv],s_conv.axis[2], "LB")
            WB = s.reuse_at(LB,s[s_conv],s_conv.axis[3], "WB")
            # print(hcl.lower(s))
            # sys.exit()
            # s[s_conv].compute_at(s[s_layer],s_layer.axis[3])
        elif layer == "maxpool2":
            s[s_layer].pipeline(s_layer.axis[2])
        elif "unpack" in layer:
            s[s_layer].pipeline(s_layer.axis[1])
        elif layer == "flatten":
            # x_out, x_in = s[s_layer].split(s_layer.axis[1], 32)
            s_pack = build_packed_bnn.pack
            # s[s_layer].compute_at(s[s_pack],s_pack.axis[1])
            s[s_pack].pipeline(s_pack.axis[1])
            s[s_layer].pipeline(s_layer.axis[1])
        elif layer == "fc1_xor":
            s[s_layer].pipeline(s_layer.axis[1])
            s_popcnt = build_packed_bnn.fc1_popcount
            # s[s_layer].compute_at(s[s_popcnt],s_popcnt.axis[2])
            s[s_popcnt].pipeline(s_popcnt.axis[1])
            s_matmal = build_packed_bnn.fc1_matmul
            s[s_matmal].pipeline(s_matmal.axis[1])
            # s[s_popcnt].compute_at(s[s_matmal],s_matmal.axis[2])
            s_bias = build_packed_bnn.fc1_bias
            s[s_bias].pipeline(s_bias.axis[1])
            s[s_matmal].compute_at(s[s_bias],s_bias.axis[1])
            s[s_matmal].pipeline(s_matmal.axis[1])
            s_fc1 = build_packed_bnn.fc1
            # s[s_bias].compute_at(s[s_fc1],s_fc1.axis[1])
            s[s_fc1].pipeline(s_fc1.axis[1])
        elif layer == "fc2_xor":
            s[s_layer].pipeline(s_layer.axis[1])
            s_popcnt = build_packed_bnn.fc2_popcount
            s[s_popcnt].pipeline(s_popcnt.axis[1])
            # s[s_layer].compute_at(s[s_popcnt],s_popcnt.axis[2])
            s_matmal = build_packed_bnn.fc2_matmul
            s[s_matmal].pipeline(s_matmal.axis[1])
            # s[s_popcnt].compute_at(s[s_matmal],s_matmal.axis[2])
            s_fc2 = build_packed_bnn.fc2
            # s[s_matmal].compute_at(s[s_fc2],s_fc2.axis[1])
            # s[s_fc2].pipeline(s_fc2.axis[1])
            s[s_fc2].pipeline(s_fc2.axis[1])

    target = "vhls"
    f = hcl.build(s, target=target)
    print(f)
    # with open("vhls_code.cpp","w") as outfile:
    #     outfile.write(f)
    # sys.exit()
    # if isinstance(target,hcl.platform):
    #     s.to([input_image] + hcl_ph, target.xcel)
    #     s.to(build_packed_bnn.fc2, target.host)
    #     target.config(compile="vivado_hls", mode="csyn")

    return hcl.build(s, target=target)

if __name__ == '__main__':

    if len(sys.argv) == 1:
        hcl_array = []
        for name in params:
            dtype = qtype_bit if ("conv" in name or "w_" in name) else qtype_float
            hcl_array.append(hcl.asarray(params[name],dtype=dtype))
        hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)
        f = build_bnn_inf()
    else:
        hcl_array = []
        for name in packed_params:
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
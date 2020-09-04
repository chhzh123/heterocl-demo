import heterocl as hcl
import hlib
import numpy as np
import sys
from bnn_brust import *

batch_size = 1
target = hcl.platform.zc706
target.config(compile="vivado_hls", mode="csyn")
# target = hcl.platform.aws_f1
# target.config(compile="vitis", mode="hw_exe", backend="vhls", project="project-vitis.prj")

hcl_array = []
hcl_image = hcl.asarray(images[:batch_size], dtype=qtype_bit)
hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)
for name in packed_params:
    if "w_conv2" in name:
        dtype = hcl.UInt(16)
    else:
        dtype = qtype_bit if "conv" in name else (qtype_packed if "w_fc" in name else qtype_float)
    hcl_array.append(hcl.asarray(packed_params[name],dtype=dtype))

f = build_bitpacked_bnn_inf_opt(batch_size,target)
print("Done building function")
f(hcl_image, *hcl_array, hcl_out)
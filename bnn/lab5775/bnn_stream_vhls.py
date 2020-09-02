import heterocl as hcl
import hlib
import numpy as np
import sys
from bnn_stream import *

batch_size = 1
target = hcl.platform.zc706
target.config(compile="vivado_hls", mode="csyn")
# target = hcl.platform.aws_f1
# target.config(compile="vitis", mode="debug") #mode="hw_exe", backend="vhls")

hcl_array = []
hcl_image = hcl.asarray(images[:batch_size], dtype=qtype_bit)
hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)
if len(sys.argv) == 1 or sys.argv[1] == 2:
	for name in params:
		dtype = qtype_bit if ("conv" in name or "w_" in name) else qtype_float
		hcl_array.append(hcl.asarray(params[name],dtype=dtype))
else:
	for name in packed_params:
		if "w_conv2" in name:
			dtype = hcl.UInt(16)
		else:
			dtype = qtype_bit if "conv" in name else (qtype_packed if "w_fc" in name else qtype_float)
		hcl_array.append(hcl.asarray(packed_params[name],dtype=dtype))

if len(sys.argv) == 1:
	f = build_bnn_inf(batch_size,target)
elif sys.argv[1] == "2":
	f = build_bnn_inf_opt(batch_size,target)
elif sys.argv[1] == "3":
	f = build_bitpacked_bnn_inf(batch_size,target)
elif sys.argv[1] == "4":
	f = build_bitpacked_bnn_inf_opt(batch_size,target)
else:
	raise RuntimeError("Not supported mode")
print("Done building function")
f(hcl_image, *hcl_array, hcl_out)
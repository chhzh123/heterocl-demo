import os, sys
from bnn_const_nhwc import *

hcl_image = hcl.asarray(images[:batch_size].transpose(0,2,3,1), dtype=dtype_in)
hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=dtype_out)

if not args.opt:
	f = build_bitpacked_bnn_inf(batch_size,target)
else:
	print("[INFO] Use optimization")
	f = build_bitpacked_bnn_inf_opt(batch_size,target)
print("Done building function")
f(hcl_image, hcl_out)
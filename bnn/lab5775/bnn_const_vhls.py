import os, sys
from bnn_const import *

hcl_image = hcl.asarray(images[:batch_size], dtype=dtype_in)
hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=dtype_out)

if not args.opt:
	f = build_bitpacked_bnn_inf(batch_size,target)
else:
	print("[INFO] Use optimization")
	f = build_bitpacked_bnn_inf_opt(batch_size,target)
print("Done building function")
f(hcl_image, hcl_out)
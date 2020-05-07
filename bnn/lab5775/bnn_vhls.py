import heterocl as hcl
import hlib
import numpy as np
from bnn_main import *

batch_size = 1

f = build_bnn_inf(batch_size,"vhls")

# add HLS pragmas manually
def add_pipeline_pad(f):
	lines = f.split("\n")
	res_f = ""
	cnt = 0
	for line in lines:
		cnt -= 1
		if line.strip()[:6] == "ap_int" and "pad" in line:
			cnt = 2
		elif cnt == 0:
			res_f += "#pragma HLS pipeline\n"
		res_f += line + "\n"
	return res_f

f = add_pipeline_pad(f)
with open("bnn.cpp","w") as outfile:
    outfile.write(f)
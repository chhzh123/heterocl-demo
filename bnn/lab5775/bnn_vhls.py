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

def add_array_reshape(f):
	res_f = ""
	lines = f.split("\n")
	for i,line in enumerate(lines):
		if "default_function" in line:
			break
	pragmas = []
	for var in ["input_image",
				"w_conv1","bn_t1",
				"w_conv2","bn_t2",
				"w_fc1","b_fc1",
				"w_fc2","b_fc2",
				"fc2"]:
		if var in ["bn_t1","w_conv2","bn_t2","w_fc1","b_fc1","w_fc2"]:
			pragmas.append("#pragma HLS ARRAY_RESHAPE variable={} block factor=256 dim=1".format(var))
		else:
			pragmas.append("#pragma HLS ARRAY_RESHAPE variable={} complete dim=1".format(var))
	lines = lines[:i+1] + pragmas + lines[i+1:]
	res_f += "\n".join(lines)
	return res_f

f = add_pipeline_pad(f)
f = add_array_reshape(f)
with open("bnn.cpp","w") as outfile:
    outfile.write(f)
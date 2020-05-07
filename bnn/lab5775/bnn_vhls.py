import heterocl as hcl
import hlib
import numpy as np
from bnn_main import *

batch_size = 1

f = build_bnn_inf(batch_size,"vhls")

def add_loop_label(f):
	loop_name = ["pad","conv_bn1","maxpool1",
				 "pad1","conv_bn2","maxpool2",
				 "flatten","fc1","fc2"]
	lines = f.split("\n")
	cnt = 0
	res_f = ""
	for i,line in enumerate(lines):
		if line[:5] == "  for":
			res_f += "{}: {}\n".format(loop_name[cnt],line.strip())
			cnt += 1
		else:
			res_f += line + "\n"
	return res_f

# add HLS pragmas manually
def add_pipeline_pad(f):
	lines = f.split("\n")
	res_f = ""
	cnt = 0
	for line in lines:
		cnt -= 1
		if line.strip()[:6] == "ap_int" and ("pad" in line or "flatten" in line):
			cnt = 3
			name = line.strip().split()[1].split("[")[0]
		elif cnt == 2:
			res_f += "#pragma HLS ARRAY_RESHAPE variable={} block factor=32 dim=1\n".format(name)
		elif cnt == 0 and name != "flatten":
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
		if var not in ["b_fc2","fc2"]:
			pragmas.append("#pragma HLS ARRAY_RESHAPE variable={} block factor=32 dim=1".format(var))
		else:
			pragmas.append("#pragma HLS ARRAY_RESHAPE variable={} complete dim=1".format(var))
	lines = lines[:i+1] + pragmas + lines[i+1:]
	res_f += "\n".join(lines)
	return res_f

f = add_loop_label(f)
f = add_pipeline_pad(f)
f = add_array_reshape(f)
with open("bnn.cpp","w") as outfile:
    outfile.write(f)
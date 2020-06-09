import heterocl as hcl
import hlib
import numpy as np
import sys
from bnn_main import *

batch_size = 1
target = hcl.platform.zc706

# add HLS pragmas manually
def add_array_partition(f):
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
			pragmas.append("#pragma HLS array_partition variable={} block factor=8 dim=1".format(var))
		else:
			pragmas.append("#pragma HLS array_partition variable={} complete dim=1".format(var))
	lines = lines[:i+1] + pragmas + lines[i+1:]
	res_f += "\n".join(lines)
	return res_f

hcl_array = []
hcl_image = hcl.asarray(images[:batch_size], dtype=qtype_bit)
hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)
if len(sys.argv) == 1 or sys.argv[1] == 2:
	for name in params:
		dtype = qtype_bit if ("conv" in name or "w_" in name) else qtype_float
		hcl_array.append(hcl.asarray(params[name],dtype=dtype))
else:
	for name in packed_params:
		dtype = qtype_bit if "conv" in name else (qtype_packed if "w_fc" in name else qtype_float)
		hcl_array.append(hcl.asarray(packed_params[name],dtype=dtype))

def parse_report():
	# hcl.report.parse_xml("project",True)
	report = f.report(target)
	overall = 0
	loop_num = open("project/kernel.cpp","r").read().count("LOOP")
	for i in range(10,loop_num):
		try:
			latency = report["PerformanceEstimates"]["SummaryOfLoopLatency"]["LOOP{}".format(i)]["Latency"]
		except:
			try:
				latency = report["PerformanceEstimates"]["SummaryOfLoopLatency"]["Loop{}".format(i)]["Latency"]
			except:
				latency = report["PerformanceEstimates"]["SummaryOfLoopLatency"]["LOOP{}_L".format(i)]["Latency"]
		print("Loop {}: {}".format(i,latency))
		overall += int(latency)
	print("Overall: {}".format(overall))

if len(sys.argv) == 1:
	f = build_bnn_inf(batch_size,target)
	f(hcl_image, *hcl_array, hcl_out)
	parse_report()
elif sys.argv[1] == "2":
	f = build_bnn_inf_opt(batch_size,target)
	# f = add_array_partition(f)
	f(hcl_image, *hcl_array, hcl_out)
	parse_report()
elif sys.argv[1] == "3":
	f = build_bitpacked_bnn_inf(batch_size,target)
	f(hcl_image, *hcl_array, hcl_out)
	parse_report()
elif sys.argv[1] == "4":
	f = build_bitpacked_bnn_inf_opt(batch_size,target)
	f(hcl_image, *hcl_array, hcl_out)
	parse_report()
else:
	raise RuntimeError("Not supported mode")
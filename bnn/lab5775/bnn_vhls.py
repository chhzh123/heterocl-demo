import heterocl as hcl
import hlib
import numpy as np
import sys
from bnn_main import *

batch_size = 1
target = hcl.platform.zc706
target.config(compile="vivado_hls", mode="csyn")

hcl_array = []
hcl_image = hcl.asarray(images[:batch_size], dtype=qtype_bit)
hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)
if len(sys.argv) == 1 or sys.argv[1] == 2:
	for name in params:
		dtype = qtype_bit if ("conv" in name or "w_" in name) else qtype_float
		hcl_array.append(hcl.asarray(params[name],dtype=dtype))
else:
	for name in packed_params:
		if "w_conv2" in name and PACK_CONV:
			dtype = hcl.UInt(16)
		else:
			dtype = qtype_bit if "conv" in name else (qtype_packed if "w_fc" in name else qtype_float)
		hcl_array.append(hcl.asarray(packed_params[name],dtype=dtype))

def parse_report(host_flag=True):
	path = "project" if host_flag else ""
	report = hcl.report.parse_xml(path,True)
	# report = f.report(target)
	overall = 0
	loop_num = open("project/kernel.cpp" if host_flag else "vhls_code.cpp","r").read().count("LOOP")
	beg = 10 if host_flag else 1
	end = loop_num if host_flag else loop_num + 1
	for i in range(beg,end):
		try:
			loop = report["PerformanceEstimates"]["SummaryOfLoopLatency"]["LOOP{}".format(i)]
		except:
			try:
				loop = report["PerformanceEstimates"]["SummaryOfLoopLatency"]["Loop{}".format(i)]
			except:
				loop = report["PerformanceEstimates"]["SummaryOfLoopLatency"]["LOOP{}_L".format(i)]
		latency = loop["Latency"]
		II = loop["PipelineII"]
		print("Loop {}: {}\tII: {}".format(i,latency,II))
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
	# f = build_bitpacked_bnn_inf_opt(batch_size,target)
	# f(hcl_image, *hcl_array, hcl_out)
	parse_report(False)
else:
	raise RuntimeError("Not supported mode")
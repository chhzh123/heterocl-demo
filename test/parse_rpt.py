import os

def extract_rpt():
	res = ""
	rptfile = os.path.join("project", "out.prj", "solution1/syn/report/test_csynth.rpt")
	with open(rptfile, "r") as rpt:
		rpt = list(map(str.lstrip,list(rpt)))
		for i, line in enumerate(rpt):
			if "Timing (ns)" in line:
				res += "* Timing (ns)\n  "
				res += '  '.join(rpt[i+2:i+7])
			elif "Latency (clock cycles)" in line:
				res += "* Latency (clock cycles)\n  "
				res += '  '.join(rpt[i+2:i+8])
			elif "Utilization Estimates" in line:
				res += "* Utilization Estimates\n  "
				res += '  '.join(rpt[i+3:i+20])
	return res

print(extract_rpt())
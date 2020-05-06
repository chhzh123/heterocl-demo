import os
import sys
import datetime

prj = sys.argv[1]
time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
outfile = open("results.txt","a+")
outfile.write("{} {}\n".format(time,prj))
with open(os.path.join(prj, prj.split("/")[0]+"_prj", "solution/syn/report/default_function_csynth.rpt"), "r") as rpt:
	rpt = list(rpt)
	for i, line in enumerate(rpt):
		if "Timing (ns)" in line:
			outfile.write(''.join(rpt[i+2:i+7]))
		elif "Latency (clock cycles)" in line:
			outfile.write(''.join(rpt[i+2:i+8]))
		elif "Utilization Estimates" in line:
			outfile.write(''.join(rpt[i+3:i+20]))
outfile.write("\n")
outfile.close()
import os
import json
import xmltodict
from tabulate import tabulate

def extract_rpt():
    report = {}
    rptfile = os.path.join("project", "out.prj", "solution1/syn/report/test_csynth.rpt")
    with open(rptfile, "r") as rpt:
        rpt = list(map(str.lstrip,list(rpt)))
        for i, line in enumerate(rpt):
            if "Version" in line:
                report["Version"] = line.split()[2]
            elif "Product family" in line:
                report["Product family"] = line.split()[2]
            elif "Target device" in line:
                report["Target device"] = line.split()[2]
            elif "Timing (ns)" in line:
                report["Timing"] = {}
                _, _, target, estimated, uncertainty = rpt[i+5].split("|")
                report["Timing"]["Unit"] = "ns"
                report["Timing"]["Target"] = target.strip()
                report["Timing"]["Estimated"] = estimated.strip()
                report["Timing"]["Uncertainty"] = uncertainty.strip()
            elif "Latency (clock cycles)" in line:
                report["Latency"]["Unit"] = "clock cycles"
                _, min_l, max_l, min_i, max_i, pipeline_type = rpt[i+6].split("|")
                report["Latency"]["MinLatency"] = min_l
                report["Latency"]["MaxLatency"] = max_l
                report["Latency"]["MinInterval"] = min_i
                report["Latency"]["MaxInterval"] = max_i
    return report

def extract_xml():
    outfile = open("project/profile.json","w")
    with open(os.path.join("project", "out.prj", "solution1/syn/report/test_csynth.xml"),"r") as xml:
        profile = xmltodict.parse(xml.read())["profile"]
        json.dump(profile,outfile,indent=2)
    res = {}
    res["HLS Version"] = "Vivado HLS " + profile["ReportVersion"]["Version"]
    res["Product family"] = profile["UserAssignments"]["ProductFamily"]
    res["Target device"] = profile["UserAssignments"]["Part"]
    clock_unit = profile["UserAssignments"]["unit"]
    res["Top Model Name"] = profile["UserAssignments"]["TopModelName"]
    res["Target Clock Period"] = profile["UserAssignments"]["TargetClockPeriod"] + clock_unit
    res["Latency"] = "Min {:>10} cycles\n".format(profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Best-caseLatency"]) + \
                     "Max {:>10} cycles".format(profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["Worst-caseLatency"])
    est_resources = profile["AreaEstimates"]["Resources"]
    avail_resources = profile["AreaEstimates"]["AvailableResources"]
    resources = {}
    for name in ["BRAM_18K","DSP48E","FF","LUT"]:
        item = [est_resources[name], avail_resources[name]]
        item.append("{}%".format(round(int(item[0])/int(item[1])*100)))
        resources[name] = item.copy()
    res["Resources"] = tabulate([[key] + resources[key] for key in resources.keys()],
                                headers=["Name","Total","Available","Utilization"])
    lst = list(res.items())
    tablestr = tabulate(lst,tablefmt="psql").split("\n")
    endash = tablestr[0].split("+")
    splitline = "+" + endash[1] + "+" + endash[2] + "+"
    tablestr.insert(5,splitline)
    table = '\n'.join(tablestr)
    print(table)
    return profile

profile = extract_xml()
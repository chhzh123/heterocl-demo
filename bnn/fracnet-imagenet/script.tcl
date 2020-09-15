############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project model-new.prj -reset
set_top FracNet
add_files pgconv.cc
add_files net_hls.h
add_files net_hls.cc
add_files matmul.cc
add_files biconv.cc
#add_files -tb tb.cc -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xczu3eg-sbva484-1-e}
create_clock -period 4 -name default
config_sdx -target none
#config_export -format ip_catalog -rtl verilog -vivado_optimization_level 2 -vivado_phys_opt place -vivado_report_level 0
set_clock_uncertainty 12.5%
#source "./model/solution1/directives.tcl"
#csim_design -ldflags {-Wl,--stack,10485670} -clean
csynth_design
#cosim_design -ldflags {-Wl,--stack,10485670}
export_design -rtl verilog -format ip_catalog

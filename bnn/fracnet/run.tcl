############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################

open_project fast.prj -reset

set_top ResNet

add_files biconv.cc
add_files matmul.cc
add_files net_hls.cc
add_files net_hls.h
add_files pgconv64.h
add_files weights.h

open_solution "solution1"
set_part {xczu3eg-sbva484-1-e}

create_clock -period 5 -name default

#csim_design
csynth_design
#cosim_design
export_design -rtl verilog -format ip_catalog
# run.tcl
# set_param general.maxThreads 2

# open the HLS project
set src_dir "."
open_project -reset knndigit_prj

# set the top-level function of the design
set_top default_function

# add design and testbench files
# add_files $src_dir/knn-digit.h
add_files $src_dir/knn-digit.cpp
# add_files -tb $src_dir/knn-digit_test.cpp

open_solution "solution"

# use Zynq device
# set_part {xc7z020clg484-1}
# use Xilinx Virtex UltraScale+ VU9P FPGA
set_part {xcvu9p-fsgd2104-3-e}

# target clock period is 4 ns (250MHz)
create_clock -period 4 -name default

# do a c simulation
# csim_design -clean

# synthesize the design
csynth_design

# do a co-simulation
# cosim_design

# close project and quit
close_project
exit
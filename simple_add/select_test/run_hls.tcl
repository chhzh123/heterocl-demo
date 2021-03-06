# run.tcl

# open the HLS project 
set src_dir "."
open_project -reset select_test_prj

# set the top-level function of the design
set_top default_function

# add design and testbench files
# add_files $src_dir/select_test.h
add_files $src_dir/select_test.cpp
# add_files -tb $src_dir/select_test_test.cpp

open_solution "solution"

# use Zynq device
set_part {xc7z020clg484-1}

# target clock period is 10 ns
create_clock -period 10 -name default

# do a c simulation
# csim_design -clean

# synthesize the design
csynth_design

# do a co-simulation
# cosim_design

# close project and quit
close_project
exit
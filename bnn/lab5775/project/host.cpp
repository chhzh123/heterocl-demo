
#include <sys/ipc.h>
#include <sys/shm.h>

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <cassert>

// opencl harness headers
#include "xcl2.hpp"
#include "ap_fixed.h"
#include "ap_int.h"
#include <cmath>
#include <vector>


int main(int argc, char ** argv) {
  std::cout << " Initialize shared memory...\n";
  uint8_t* arg_0 = (uint8_t*)shmat(/*input_image*/31457736, nullptr, 0);
  auto input_image = new ap_uint<8>[1][16][16][1];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        for (size_t i3 = 0; i3 < 1; i3++) {
          input_image[i0][i1][i2][i3] = 0; //(uint8_t)(arg_0[i3 + i2*1 + i1*16 + i0*256]);
        }
      }
    }
  }

  int32_t* arg_1 = (int32_t*)shmat(/*fc2*/31490506, nullptr, 0);
  auto fc2 = new ap_fixed<32,22>[1][10];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      fc2[i0][i1] = 0; //(int32_t)(arg_1[i1 + i0*10]) >> 10;
    }
  }

  std::cout << " Initialize RTE...\n";

  if (argc != 2) {
      std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
      return EXIT_FAILURE;
  }

  auto binaryFile = argv[1];
  cl_int err = CL_SUCCESS;

  // create binary file and program
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

  cl::Context context;
  cl::CommandQueue q;
  cl::Program program;
  auto devices = xcl::get_xil_devices();
  int valid_device = 0;

  for (unsigned int i = 0; i < devices.size(); i++) {
      auto device = devices[i];
      // Creating Context and Command Queue for selected Device
      context = cl::Context(device, NULL, NULL, NULL, &err);
      q = cl::CommandQueue(
          context, device, CL_QUEUE_PROFILING_ENABLE, &err);

      std::cout << "Trying to program device[" << i
                << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
      program = cl::Program(context, {device}, bins, NULL, &err);
      if (err != CL_SUCCESS) {
          std::cout << "Failed to program device[" << i
                    << "] with xclbin file!\n";
      } else {
          std::cout << "Device[" << i << "]: program successful!\n";
          valid_device++;
          break; // we break because we found a valid device
      }
  }
  if (valid_device == 0) {
      std::cout << "Failed to program any device found, exit!\n";
      exit(EXIT_FAILURE);
  }


  // compute and kernel call from host
  ap_int<32> __device_scope;

  cl::Kernel kernel(program, "test", &err);
  cl::Buffer buffer_input_image(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_uint<8>)*1*16*16*1, input_image, &err);
  cl::Buffer buffer_fc2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_fixed<32, 22>)*1*10, fc2, &err);

  // set device kernel buffer
  err = kernel.setArg(0, buffer_input_image);
  err = kernel.setArg(1, buffer_fc2);
  err = q.enqueueMigrateMemObjects({buffer_input_image, buffer_fc2}, 0/*from host*/);
  q.finish();

  // enqueue kernel function
  std::chrono::duration<double> kernel_time(0);
  auto kernel_start = std::chrono::high_resolution_clock::now();
  cl::Event event;
  err = q.enqueueTask(kernel, NULL, &event);

  err = q.finish();
  auto kernel_end = std::chrono::high_resolution_clock::now();
  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
  auto kernel_time_in_sec = kernel_time.count();
  std::cout << "Execution Time:" <<  kernel_time_in_sec;
  err = q.enqueueMigrateMemObjects({buffer_input_image, buffer_fc2}, CL_MIGRATE_MEM_OBJECT_HOST);

  // execution on host 

  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        for (size_t i3 = 0; i3 < 1; i3++) {
          ;//arg_0[i3 + i2*1 + i1*16 + i0*256] = (uint8_t)(input_image[i0][i1][i2][i3]);
        }
      }
    }
  }
  shmdt(arg_0);
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      ;//arg_1[i1 + i0*10] = (int32_t)(fc2[i0][i1]) << 10;
    }
  }
  shmdt(arg_1);


  }

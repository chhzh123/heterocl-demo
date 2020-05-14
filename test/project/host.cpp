
#include <sys/ipc.h>
#include <sys/shm.h>

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <cassert>

// vivado hls headers
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include "kernel.h"

int main(int argc, char ** argv) {
  int32_t* arg_0 = (int32_t*)shmat(0, nullptr, 0);
  int32_t* A = new int32_t[10 * 32];
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      A[i1 + i0*32] = (int32_t)(arg_0[i1 + i0*32]);
    }
  }

  int32_t* arg_1 = (int32_t*)shmat(1, nullptr, 0);
  int32_t* D = new int32_t[10 * 32];
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      D[i1 + i0*32] = (int32_t)(arg_1[i1 + i0*32]);
    }
  }


  // compute and kernel call from host
  ap_int<32> _top;
  ap_int<32> B[320];
  for (ap_int<32> args = 0; args < 10; ++args) {
    for (ap_int<32> args0 = 0; args0 < 32; ++args0) {
      B[(args0 + (args * 32))] = (A[(args0 + (args * 32))] + 1);
    }
  }
  hls::stream<ap_int<32>> B_channel;
  for (ap_int<32> B0 = 0; B0 < 10; ++B0) {
    for (ap_int<32> B1 = 0; B1 < 32; ++B1) {
      B_channel.write(B[(B1 + (B0 * 32))]);
    }
  }
  hls::stream<ap_int<32>> C_channel;
  test(B_channel, C_channel);
  ap_int<32> C[320];
  for (ap_int<32> C0 = 0; C0 < 10; ++C0) {
    for (ap_int<32> C1 = 0; C1 < 32; ++C1) {
      C[(C1 + (C0 * 32))] = C_channel.read();
    }
  }
  for (ap_int<32> args1 = 0; args1 < 10; ++args1) {
    for (ap_int<32> args01 = 0; args01 < 32; ++args01) {
      D[(args01 + (args1 * 32))] = (C[(args01 + (args1 * 32))] * 2);
    }
  }

  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      arg_0[i1 + i0*32] = (int32_t)(A[i1 + i0*32]);
    }
  }
  shmdt(arg_0);
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 32; i1++) {
      arg_1[i1 + i0*32] = (int32_t)(D[i1 + i0*32]);
    }
  }
  shmdt(arg_1);


  }

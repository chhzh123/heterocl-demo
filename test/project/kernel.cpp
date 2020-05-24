#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include "kernel.h"
  void test(hls::stream<bit32>& B_channel, hls::stream<bit32>& C_channel) {
  #pragma HLS INTERFACE axis port=B_channel offset=slave bundle=gmem0
  #pragma HLS INTERFACE axis port=C_channel offset=slave bundle=gmem1
  #pragma HLS INTERFACE s_axilite port=return bundle=control
    bit32 B[320];
    for (bit32 B0 = 0; B0 < 10; ++B0) {
      for (bit32 B1 = 0; B1 < 32; ++B1) {
        B[(B1 + (B0 * 32))] = B_channel.read();
      }
    }
    bit32 C[320];
    for (bit32 args = 0; args < 10; ++args) {
      for (bit32 args0 = 0; args0 < 32; ++args0) {
        C[(args0 + (args * 32))] = (B[(args0 + (args * 32))] + 1);
      }
    }
    for (bit32 C0 = 0; C0 < 10; ++C0) {
      for (bit32 C1 = 0; C1 < 32; ++C1) {
        C_channel.write(C[(C1 + (C0 * 32))]);
      }
    }
  }

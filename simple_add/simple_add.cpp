#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

void default_function(ap_int<32>* A, ap_uint<4>* B) {
                  #pragma HLS array_partition variable=A complete dim=0
  for (ap_int<32> y = 0; y < 8; ++y) {
    for (ap_int<32> x = 0; x < 8; ++x) {
    #pragma HLS pipeline
      B[(x + (y * 8))] = ((ap_uint<4>)(((ap_int<33>)A[(x + (y * 10))]) + ((ap_int<33>)A[((x + (y * 10)) + 22)])));
    }
  }
}


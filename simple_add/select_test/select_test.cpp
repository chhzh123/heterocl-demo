#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

void default_function(ap_int<8> A[8*8], ap_fixed<16, 4> B[8*8], ap_int<32> C[8*8]) {
  ap_int<32> _top;
  for (ap_int<32> y = 0; y < 8; ++y) {
    for (ap_int<32> x = 0; x < 8; ++x) {
      C[(x + (y * 8))] = ((ap_int<32>)((x < 4) ? ((ap_fixed<32, 0>)((ap_int<16>)A[(x + (y * 8))])) : ((ap_fixed<32, 0>)((ap_int<16>)B[(x + (y * 8))]))));
    }
  }
}
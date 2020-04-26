#include <ap_int.h>
#include <ap_fixed.h>
#include <math.h>

void default_function(ap_int<32> placeholder2[320][32], ap_int<32> placeholder3[16][32], ap_int<32> compute3[320]) {
  ap_int<32> _top;
  for (ap_int<32> x = 0; x < 320; ++x) {
    compute3[x] = 0;
  }
  ap_int<32> main_loop;
  for (ap_int<32> _ = 0; _ < 200; ++_) {
    for (ap_int<32> N = 0; N < 320; ++N) {
    #pragma HLS pipeline
      ap_int<32> scalar2;
      scalar2 = 100000;
      for (ap_int<32> i = 0; i < 16; ++i) {
        ap_int<32> scalar3;
        scalar3 = 0;
        for (ap_int<32> i1 = 0; i1 < 32; ++i1) {
          scalar3 = ((ap_int<32>)(((ap_int<67>)scalar3) + ((ap_int<67>)(((ap_int<66>)((ap_int<33>)(placeholder2[N][i1] - placeholder3[i][i1]))) * ((ap_int<66>)((ap_int<33>)(placeholder2[N][i1] - placeholder3[i][i1])))))));
        }
        if (scalar3 < scalar2) {
          scalar2 = scalar3;
          compute3[N] = i;
        }
      }
    }
    ap_int<32> compute4[16];
    for (ap_int<32> x1 = 0; x1 < 16; ++x1) {
      compute4[x1] = 0;
    }
    ap_int<32> compute5[16][32];
    for (ap_int<32> x2 = 0; x2 < 16; ++x2) {
      for (ap_int<32> y = 0; y < 32; ++y) {
        compute5[x2][y] = 0;
      }
    }
    ap_int<32> calc_sum;
    for (ap_int<32> n = 0; n < 320; ++n) {
    #pragma HLS unroll
      compute4[compute3[n]] = (compute4[compute3[n]] + 1);
      for (ap_int<32> i2 = 0; i2 < 32; ++i2) {
        compute5[compute3[n]][i2] = ((ap_int<32>)(((ap_int<33>)compute5[compute3[n]][i2]) + ((ap_int<33>)placeholder2[n][i2])));
      }
    }
    ap_int<32> update_mean;
    for (ap_int<32> k_d_fused = 0; k_d_fused < 512; ++k_d_fused) {
    #pragma HLS unroll
      placeholder3[(k_d_fused / 32)][(k_d_fused % 32)] = (compute5[(k_d_fused / 32)][(k_d_fused % 32)] / compute4[(k_d_fused / 32)]);
    }
  }
}
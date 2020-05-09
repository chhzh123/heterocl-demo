#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

void default_function(ap_uint<1> input_image[1*1*16*16], ap_uint<1> w_conv1[16*1*3*3], ap_fixed<20, 10> bn_t1[16*16*16], ap_uint<1> w_conv2[32*16*3*3], ap_fixed<20, 10> bn_t2[32*8*8], ap_uint<1> w_fc1[256*512], ap_fixed<20, 10> b_fc1[256], ap_uint<1> w_fc2[10*256], ap_fixed<20, 10> b_fc2[10], ap_fixed<20, 10> fc2[1*10]) {
#pragma HLS ARRAY_RESHAPE variable=input_image block factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=w_conv1 block factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=bn_t1 block factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=w_conv2 block factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=bn_t2 block factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=w_fc1 block factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=b_fc1 block factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=w_fc2 block factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=b_fc2 complete dim=1
#pragma HLS ARRAY_RESHAPE variable=fc2 complete dim=1
  ap_int<32> _top;
  ap_uint<1> pad[324];
LOOP_PAD: for (ap_int<32> index_tuple = 0; index_tuple < 18; ++index_tuple) {
    for (ap_int<32> i = 0; i < 18; ++i) {
      pad[(i + (index_tuple * 18))] = (((((1 <= index_tuple) && (index_tuple < 17)) && (1 <= i)) && (i < 17)) ? ((ap_uint<1>)input_image[((i + (index_tuple * 16)) + -17)]) : ((ap_uint<1>)0));
    }
  }
  ap_uint<1> bn1[4096];
LOOP_CONV_BN1: for (ap_int<32> c = 0; c < 16; ++c) {
    for (ap_int<32> h = 0; h < 16; ++h) {
      for (ap_int<32> w = 0; w < 16; ++w) {
      #pragma HLS pipeline
        ap_int<6> conv1;
        ap_int<6> sum;
        sum = (ap_int<6>)0;
        for (ap_int<32> ry = 0; ry < 3; ++ry) {
          for (ap_int<32> rx = 0; rx < 3; ++rx) {
            sum = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)rx)) <= ((ap_int<33>)w)) && (((ap_int<33>)w) < ((ap_int<33>)17 - ((ap_int<33>)rx)))) && (((ap_int<33>)1 - ((ap_int<33>)ry)) <= ((ap_int<33>)h))) && (((ap_int<33>)h) < ((ap_int<33>)17 - ((ap_int<33>)ry)))) ? ((ap_uint<32>)((((ap_uint<32>)(pad[((w + rx) + ((h + ry) * 18))] == w_conv1[((rx + (ry * 3)) + (c * 9))])) * 2U) - 1U)) : (0U))) + ((ap_int<34>)sum)));
          }
        }
        conv1 = sum;
        bn1[((w + (h * 16)) + (c * 256))] = ((ap_uint<1>)((bn_t1[((w + (h * 16)) + (c * 256))] < ((ap_fixed<20, 10>)conv1)) ? 1 : 0));
      }
    }
  }
  ap_uint<1> maxpool1[1024];
LOOP_MAXPOOL1: for (ap_int<32> i1 = 0; i1 < 1; ++i1) {
    for (ap_int<32> c1 = 0; c1 < 16; ++c1) {
      for (ap_int<32> h1 = 0; h1 < 8; ++h1) {
      #pragma HLS pipeline
        for (ap_int<32> w1 = 0; w1 < 8; ++w1) {
          ap_uint<1> reducer4;
          reducer4 = (ap_uint<1>)0;
          for (ap_int<32> ra6 = 0; ra6 < 2; ++ra6) {
            for (ap_int<32> ra7 = 0; ra7 < 2; ++ra7) {
              reducer4 = std::max(bn1[(((((w1 * 2) + ra7) + (((h1 * 2) + ra6) * 16)) + (c1 * 256)) + (i1 * 4096))], reducer4);
            }
          }
          maxpool1[(((w1 + (h1 * 8)) + (c1 * 64)) + (i1 * 1024))] = ((ap_uint<1>)((0U < ((ap_uint<32>)reducer4)) ? 1 : 0));
        }
      }
    }
  }
  ap_uint<1> pad1[1600];
LOOP_PAD1: for (ap_int<32> not_zero = 0; not_zero < 16; ++not_zero) {
    for (ap_int<32> index_tuple1 = 0; index_tuple1 < 10; ++index_tuple1) {
      for (ap_int<32> i2 = 0; i2 < 10; ++i2) {
        pad1[((i2 + (index_tuple1 * 10)) + (not_zero * 100))] = (((((1 <= index_tuple1) && (index_tuple1 < 9)) && (1 <= i2)) && (i2 < 9)) ? ((ap_uint<1>)maxpool1[(((i2 + (index_tuple1 * 8)) + (not_zero * 64)) + -9)]) : ((ap_uint<1>)0));
      }
    }
  }
  ap_uint<1> bn2[2048];
LOOP_CONV_BN2: for (ap_int<32> c2 = 0; c2 < 32; ++c2) {
    for (ap_int<32> h2 = 0; h2 < 8; ++h2) {
      for (ap_int<32> w2 = 0; w2 < 8; ++w2) {
        ap_int<6> conv2;
        ap_int<6> sum1;
        sum1 = (ap_int<6>)0;
        for (ap_int<32> rc = 0; rc < 16; ++rc) {
        #pragma HLS pipeline
          for (ap_int<32> ry1 = 0; ry1 < 3; ++ry1) {
            for (ap_int<32> rx1 = 0; rx1 < 3; ++rx1) {
              sum1 = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)rx1)) <= ((ap_int<33>)w2)) && (((ap_int<33>)w2) < ((ap_int<33>)9 - ((ap_int<33>)rx1)))) && (((ap_int<33>)1 - ((ap_int<33>)ry1)) <= ((ap_int<33>)h2))) && (((ap_int<33>)h2) < ((ap_int<33>)9 - ((ap_int<33>)ry1)))) ? ((ap_uint<32>)((((ap_uint<32>)(pad1[(((w2 + rx1) + ((h2 + ry1) * 10)) + (rc * 100))] == w_conv2[(((rx1 + (ry1 * 3)) + (rc * 9)) + (c2 * 144))])) * 2U) - 1U)) : (0U))) + ((ap_int<34>)sum1)));
            }
          }
        }
        conv2 = sum1;
        bn2[((w2 + (h2 * 8)) + (c2 * 64))] = ((ap_uint<1>)((bn_t2[((w2 + (h2 * 8)) + (c2 * 64))] < ((ap_fixed<20, 10>)conv2)) ? 1 : 0));
      }
    }
  }
  ap_uint<1> maxpool2[512];
LOOP_MAXPOOL2: for (ap_int<32> i3 = 0; i3 < 1; ++i3) {
    for (ap_int<32> c3 = 0; c3 < 32; ++c3) {
      for (ap_int<32> h3 = 0; h3 < 4; ++h3) {
      #pragma HLS pipeline
        for (ap_int<32> w3 = 0; w3 < 4; ++w3) {
          ap_uint<1> reducer5;
          reducer5 = (ap_uint<1>)0;
          for (ap_int<32> ra8 = 0; ra8 < 2; ++ra8) {
            for (ap_int<32> ra9 = 0; ra9 < 2; ++ra9) {
              reducer5 = std::max(bn2[(((((w3 * 2) + ra9) + (((h3 * 2) + ra8) * 8)) + (c3 * 64)) + (i3 * 2048))], reducer5);
            }
          }
          maxpool2[(((w3 + (h3 * 4)) + (c3 * 16)) + (i3 * 512))] = ((ap_uint<1>)((0U < ((ap_uint<32>)reducer5)) ? 1 : 0));
        }
      }
    }
  }
  ap_uint<1> flatten[512];
LOOP_FLATTEN: for (ap_int<32> i4 = 0; i4 < 1; ++i4) {
    for (ap_int<32> j = 0; j < 512; ++j) {
    #pragma HLS pipeline
      flatten[(j + (i4 * 512))] = maxpool2[(((((j / 128) * 4) + ((j / 32) % 4)) + ((j % 32) * 16)) + (i4 * 512))];
    }
  }
  ap_uint<1> dense_relu[256];
LOOP_FC1: for (ap_int<32> i5 = 0; i5 < 1; ++i5) {
    for (ap_int<32> j1 = 0; j1 < 256; ++j1) {
      ap_fixed<20, 10> fc1;
      ap_fixed<20, 10> reducer6;
      reducer6 = ((ap_fixed<20, 10>)0);
      for (ap_int<32> ra10 = 0; ra10 < 512; ++ra10) {
      #pragma HLS pipeline
        reducer6 = ((ap_fixed<20, 10>)(((ap_fixed<22, 12>)(flatten[(ra10 + (i5 * 512))] == w_fc1[(ra10 + (j1 * 512))])) + ((ap_fixed<22, 12>)reducer6)));
      }
      fc1 = ((ap_fixed<20, 10>)((((float)(((ap_fixed<53, 43>)(((ap_fixed<52, 42>)reducer6) * (ap_int<52>)2)) + (ap_fixed<53, 43>)-512)) * 6.250000e-02f) + ((float)b_fc1[j1])));
      dense_relu[(j1 + (i5 * 256))] = ((ap_uint<1>)(((ap_fixed<42, 32>)0 < ((ap_fixed<42, 32>)fc1)) ? 1 : 0));
    }
  }
LOOP_FC2: for (ap_int<32> i6 = 0; i6 < 1; ++i6) {
    for (ap_int<32> j2 = 0; j2 < 10; ++j2) {
      ap_fixed<20, 10> reducer7;
      reducer7 = ((ap_fixed<20, 10>)0);
      for (ap_int<32> ra11 = 0; ra11 < 256; ++ra11) {
      #pragma HLS pipeline
        reducer7 = ((ap_fixed<20, 10>)(((ap_fixed<22, 12>)(dense_relu[(ra11 + (i6 * 256))] == w_fc2[(ra11 + (j2 * 256))])) + ((ap_fixed<22, 12>)reducer7)));
      }
      fc2[(j2 + (i6 * 10))] = ((ap_fixed<20, 10>)((((float)(((ap_fixed<53, 43>)(((ap_fixed<52, 42>)reducer7) * (ap_int<52>)2)) + (ap_fixed<53, 43>)-256)) * 8.838835e-02f) + ((float)b_fc2[j2])));
    }
  }
}




#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

void default_function(ap_int<10> input_image[100*1*16*16], ap_uint<1> w_conv1[16*1*3*3], float bn_t1[16*16*16], ap_uint<1> w_conv2[32*16*3*3], float bn_t2[32*8*8], ap_uint<1> w_fc1[256*512], float b_fc1[256], ap_uint<1> w_fc2[10*256], float b_fc2[10], float fc2[100*10]) {
  ap_int<32> _top;
  ap_int<10> pad_temp[32400];
  for (ap_int<32> indices = 0; indices < 100; ++indices) {
    for (ap_int<32> index_tuple = 0; index_tuple < 18; ++index_tuple) {
      for (ap_int<32> i = 0; i < 18; ++i) {
        pad_temp[((i + (index_tuple * 18)) + (indices * 324))] = (((((1 <= index_tuple) && (index_tuple < 17)) && (1 <= i)) && (i < 17)) ? ((ap_int<10>)input_image[(((i + (index_tuple * 16)) + (indices * 256)) + -17)]) : ((ap_int<10>)(ap_int<10>)0));
      }
    }
  }
  ap_int<10> bn1[409600];
  for (ap_int<32> i1 = 0; i1 < 100; ++i1) {
    for (ap_int<32> c = 0; c < 16; ++c) {
      for (ap_int<32> h = 0; h < 16; ++h) {
        for (ap_int<32> w = 0; w < 16; ++w) {
        #pragma HLS pipeline
          ap_int<10> conv1;
          ap_int<10> sum;
          sum = (ap_int<10>)0;
          for (ap_int<32> ry = 0; ry < 3; ++ry) {
          #pragma HLS unroll
            for (ap_int<32> rx = 0; rx < 3; ++rx) {
              sum = ((ap_int<10>)(((ap_int<33>)(((((((ap_int<33>)1 - ((ap_int<33>)rx)) <= ((ap_int<33>)w)) && (((ap_int<33>)w) < ((ap_int<33>)17 - ((ap_int<33>)rx)))) && (((ap_int<33>)1 - ((ap_int<33>)ry)) <= ((ap_int<33>)h))) && (((ap_int<33>)h) < ((ap_int<33>)17 - ((ap_int<33>)ry)))) ? ((ap_int<32>)(((1 - ((ap_int<32>)(pad_temp[(((w + rx) + ((h + ry) * 18)) + (i1 * 324))] ^ w_conv1[((rx + (ry * 3)) + (c * 9))]))) << 1) + -1)) : ((ap_int<32>)0))) + ((ap_int<33>)sum)));
            }
          }
          conv1 = sum;
          bn1[(((w + (h * 16)) + (c * 256)) + (i1 * 4096))] = ((ap_int<10>)((bn_t1[((w + (h * 16)) + (c * 256))] < ((float)conv1)) ? 1 : 0));
        }
      }
    }
  }
  ap_int<10> pad[409600];
  for (ap_int<32> indices1 = 0; indices1 < 100; ++indices1) {
    for (ap_int<32> not_zero = 0; not_zero < 16; ++not_zero) {
      for (ap_int<32> index_tuple1 = 0; index_tuple1 < 16; ++index_tuple1) {
        for (ap_int<32> i2 = 0; i2 < 16; ++i2) {
          pad[(((i2 + (index_tuple1 * 16)) + (not_zero * 256)) + (indices1 * 4096))] = bn1[(((i2 + (index_tuple1 * 16)) + (not_zero * 256)) + (indices1 * 4096))];
        }
      }
    }
  }
  ap_int<10> maxpool1[102400];
  for (ap_int<32> i3 = 0; i3 < 100; ++i3) {
    for (ap_int<32> c1 = 0; c1 < 16; ++c1) {
      for (ap_int<32> h1 = 0; h1 < 8; ++h1) {
        for (ap_int<32> w1 = 0; w1 < 8; ++w1) {
        #pragma HLS pipeline
          ap_int<10> reducer4;
          reducer4 = (ap_int<10>)-512;
          for (ap_int<32> ra6 = 0; ra6 < 2; ++ra6) {
            for (ap_int<32> ra7 = 0; ra7 < 2; ++ra7) {
            #pragma HLS unroll
              reducer4 = std::max(pad[(((((w1 * 2) + ra7) + (((h1 * 2) + ra6) * 16)) + (c1 * 256)) + (i3 * 4096))], reducer4);
            }
          }
          maxpool1[(((w1 + (h1 * 8)) + (c1 * 64)) + (i3 * 1024))] = ((ap_int<10>)((0 < ((ap_int<32>)reducer4)) ? 1 : 0));
        }
      }
    }
  }
  ap_int<10> pad_temp1[160000];
  for (ap_int<32> indices2 = 0; indices2 < 100; ++indices2) {
    for (ap_int<32> not_zero1 = 0; not_zero1 < 16; ++not_zero1) {
      for (ap_int<32> index_tuple2 = 0; index_tuple2 < 10; ++index_tuple2) {
        for (ap_int<32> i4 = 0; i4 < 10; ++i4) {
          pad_temp1[(((i4 + (index_tuple2 * 10)) + (not_zero1 * 100)) + (indices2 * 1600))] = (((((1 <= index_tuple2) && (index_tuple2 < 9)) && (1 <= i4)) && (i4 < 9)) ? ((ap_int<10>)maxpool1[((((i4 + (index_tuple2 * 8)) + (not_zero1 * 64)) + (indices2 * 1024)) + -9)]) : ((ap_int<10>)(ap_int<10>)0));
        }
      }
    }
  }
  ap_int<10> bn2[204800];
  for (ap_int<32> i5 = 0; i5 < 100; ++i5) {
    for (ap_int<32> c2 = 0; c2 < 32; ++c2) {
      for (ap_int<32> h2 = 0; h2 < 8; ++h2) {
        for (ap_int<32> w2 = 0; w2 < 8; ++w2) {
        #pragma HLS pipeline
          ap_int<10> conv2;
          ap_int<10> sum1;
          sum1 = (ap_int<10>)0;
          for (ap_int<32> rc = 0; rc < 16; ++rc) {
            for (ap_int<32> ry1 = 0; ry1 < 3; ++ry1) {
            #pragma HLS unroll
              for (ap_int<32> rx1 = 0; rx1 < 3; ++rx1) {
                sum1 = ((ap_int<10>)(((ap_int<33>)(((((((ap_int<33>)1 - ((ap_int<33>)rx1)) <= ((ap_int<33>)w2)) && (((ap_int<33>)w2) < ((ap_int<33>)9 - ((ap_int<33>)rx1)))) && (((ap_int<33>)1 - ((ap_int<33>)ry1)) <= ((ap_int<33>)h2))) && (((ap_int<33>)h2) < ((ap_int<33>)9 - ((ap_int<33>)ry1)))) ? ((ap_int<32>)(((1 - ((ap_int<32>)(pad_temp1[((((w2 + rx1) + ((h2 + ry1) * 10)) + (rc * 100)) + (i5 * 1600))] ^ w_conv2[(((rx1 + (ry1 * 3)) + (rc * 9)) + (c2 * 144))]))) << 1) + -1)) : ((ap_int<32>)0))) + ((ap_int<33>)sum1)));
              }
            }
          }
          conv2 = sum1;
          bn2[(((w2 + (h2 * 8)) + (c2 * 64)) + (i5 * 2048))] = ((ap_int<10>)((bn_t2[((w2 + (h2 * 8)) + (c2 * 64))] < ((float)conv2)) ? 1 : 0));
        }
      }
    }
  }
  ap_int<10> pad1[204800];
  for (ap_int<32> indices3 = 0; indices3 < 100; ++indices3) {
    for (ap_int<32> not_zero2 = 0; not_zero2 < 32; ++not_zero2) {
      for (ap_int<32> index_tuple3 = 0; index_tuple3 < 8; ++index_tuple3) {
        for (ap_int<32> i6 = 0; i6 < 8; ++i6) {
          pad1[(((i6 + (index_tuple3 * 8)) + (not_zero2 * 64)) + (indices3 * 2048))] = bn2[(((i6 + (index_tuple3 * 8)) + (not_zero2 * 64)) + (indices3 * 2048))];
        }
      }
    }
  }
  ap_int<10> flatten[51200];
  for (ap_int<32> i7 = 0; i7 < 100; ++i7) {
    ap_int<10> maxpool2[512];
    for (ap_int<32> c3 = 0; c3 < 32; ++c3) {
      for (ap_int<32> h3 = 0; h3 < 4; ++h3) {
        for (ap_int<32> w3 = 0; w3 < 4; ++w3) {
        #pragma HLS pipeline
          ap_int<10> reducer5;
          reducer5 = (ap_int<10>)-512;
          for (ap_int<32> ra8 = 0; ra8 < 2; ++ra8) {
            for (ap_int<32> ra9 = 0; ra9 < 2; ++ra9) {
            #pragma HLS unroll
              reducer5 = std::max(pad1[(((((w3 * 2) + ra9) + (((h3 * 2) + ra8) * 8)) + (c3 * 64)) + (i7 * 2048))], reducer5);
            }
          }
          maxpool2[(((h3 * 4) + w3) + (c3 * 16))] = ((ap_int<10>)((0 < ((ap_int<32>)reducer5)) ? 1 : 0));
        }
      }
    }
    for (ap_int<32> j = 0; j < 512; ++j) {
      flatten[(j + (i7 * 512))] = maxpool2[((((j / 128) * 4) + ((j / 32) % 4)) + ((j % 32) * 16))];
    }
  }
  ap_int<10> dense_relu[25600];
  for (ap_int<32> i8 = 0; i8 < 100; ++i8) {
    for (ap_int<32> j1 = 0; j1 < 256; ++j1) {
      float fc1;
      float reducer6;
      reducer6 = 0.000000e+00f;
      for (ap_int<32> ra10 = 0; ra10 < 512; ++ra10) {
      #pragma HLS pipeline
        reducer6 = (((float)(((1 - ((ap_int<32>)(flatten[(ra10 + (i8 * 512))] ^ w_fc1[(ra10 + (j1 * 512))]))) << 1) + -1)) + reducer6);
      }
      fc1 = ((reducer6 * 6.250000e-02f) + b_fc1[j1]);
      dense_relu[(j1 + (i8 * 256))] = ((ap_int<10>)((0.000000e+00f < fc1) ? 1 : 0));
    }
  }
  for (ap_int<32> i9 = 0; i9 < 100; ++i9) {
    for (ap_int<32> j2 = 0; j2 < 10; ++j2) {
      float reducer7;
      reducer7 = 0.000000e+00f;
      for (ap_int<32> ra11 = 0; ra11 < 256; ++ra11) {
      #pragma HLS pipeline
        reducer7 = (((float)(((1 - ((ap_int<32>)(dense_relu[(ra11 + (i9 * 256))] ^ w_fc2[(ra11 + (j2 * 256))]))) << 1) + -1)) + reducer7);
      }
      fc2[(j2 + (i9 * 10))] = ((reducer7 * 8.838835e-02f) + b_fc2[j2]);
    }
  }
}


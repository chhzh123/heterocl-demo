#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include "kernel.h"
  void test(hls::stream<ap_uint<1> >& w_fc2_channel, hls::stream<ap_fixed<20, 10> >& b_fc2_channel, hls::stream<ap_fixed<20, 10> >& b_fc1_channel, hls::stream<ap_uint<1> >& w_fc1_channel, hls::stream<ap_fixed<20, 10> >& bn_t2_channel, hls::stream<ap_uint<1> >& w_conv2_channel, hls::stream<ap_fixed<20, 10> >& bn_t1_channel, hls::stream<ap_uint<1> >& w_conv1_channel, hls::stream<ap_uint<1> >& input_image_channel, hls::stream<ap_fixed<20, 10> >& fc2_channel) {
  #pragma HLS INTERFACE axis port=w_fc2_channel offset=slave bundle=gmem0
  #pragma HLS INTERFACE axis port=b_fc2_channel offset=slave bundle=gmem1
  #pragma HLS INTERFACE axis port=b_fc1_channel offset=slave bundle=gmem2
  #pragma HLS INTERFACE axis port=w_fc1_channel offset=slave bundle=gmem3
  #pragma HLS INTERFACE axis port=bn_t2_channel offset=slave bundle=gmem4
  #pragma HLS INTERFACE axis port=w_conv2_channel offset=slave bundle=gmem5
  #pragma HLS INTERFACE axis port=bn_t1_channel offset=slave bundle=gmem6
  #pragma HLS INTERFACE axis port=w_conv1_channel offset=slave bundle=gmem7
  #pragma HLS INTERFACE axis port=input_image_channel offset=slave bundle=gmem8
  #pragma HLS INTERFACE axis port=fc2_channel offset=slave bundle=gmem9
  #pragma HLS INTERFACE s_axilite port=return bundle=control
    ap_fixed<20, 10> fc2[10];
    ap_uint<1> w_fc2[2560];
    for (bit32 w_fc20 = 0; w_fc20 < 10; ++w_fc20) {
      for (bit32 w_fc21 = 0; w_fc21 < 256; ++w_fc21) {
        w_fc2[(w_fc21 + (w_fc20 * 256))] = w_fc2_channel.read();
      }
    }
    ap_fixed<20, 10> b_fc2[10];
    for (bit32 b_fc20 = 0; b_fc20 < 10; ++b_fc20) {
      b_fc2[b_fc20] = b_fc2_channel.read();
    }
    ap_fixed<20, 10> b_fc1[256];
    for (bit32 b_fc10 = 0; b_fc10 < 256; ++b_fc10) {
      b_fc1[b_fc10] = b_fc1_channel.read();
    }
    ap_uint<1> w_fc1[131072];
    for (bit32 w_fc10 = 0; w_fc10 < 256; ++w_fc10) {
      for (bit32 w_fc11 = 0; w_fc11 < 512; ++w_fc11) {
        w_fc1[(w_fc11 + (w_fc10 * 512))] = w_fc1_channel.read();
      }
    }
    ap_fixed<20, 10> bn_t2[2048];
    for (bit32 bn_t20 = 0; bn_t20 < 32; ++bn_t20) {
      for (bit32 bn_t21 = 0; bn_t21 < 8; ++bn_t21) {
        for (bit32 bn_t22 = 0; bn_t22 < 8; ++bn_t22) {
          bn_t2[((bn_t22 + (bn_t21 * 8)) + (bn_t20 * 64))] = bn_t2_channel.read();
        }
      }
    }
    ap_uint<1> w_conv2[4608];
    for (bit32 w_conv20 = 0; w_conv20 < 32; ++w_conv20) {
      for (bit32 w_conv21 = 0; w_conv21 < 16; ++w_conv21) {
        for (bit32 w_conv22 = 0; w_conv22 < 3; ++w_conv22) {
          for (bit32 w_conv23 = 0; w_conv23 < 3; ++w_conv23) {
            w_conv2[(((w_conv23 + (w_conv22 * 3)) + (w_conv21 * 9)) + (w_conv20 * 144))] = w_conv2_channel.read();
          }
        }
      }
    }
    ap_fixed<20, 10> bn_t1[4096];
    for (bit32 bn_t10 = 0; bn_t10 < 16; ++bn_t10) {
      for (bit32 bn_t11 = 0; bn_t11 < 16; ++bn_t11) {
        for (bit32 bn_t12 = 0; bn_t12 < 16; ++bn_t12) {
          bn_t1[((bn_t12 + (bn_t11 * 16)) + (bn_t10 * 256))] = bn_t1_channel.read();
        }
      }
    }
    ap_uint<1> w_conv1[144];
    for (bit32 w_conv10 = 0; w_conv10 < 16; ++w_conv10) {
      for (bit32 w_conv12 = 0; w_conv12 < 3; ++w_conv12) {
        for (bit32 w_conv13 = 0; w_conv13 < 3; ++w_conv13) {
          w_conv1[((w_conv13 + (w_conv12 * 3)) + (w_conv10 * 9))] = w_conv1_channel.read();
        }
      }
    }
    ap_uint<1> input_image[256];
    for (bit32 input_image2 = 0; input_image2 < 16; ++input_image2) {
      for (bit32 input_image3 = 0; input_image3 < 16; ++input_image3) {
        input_image[(input_image3 + (input_image2 * 16))] = input_image_channel.read();
      }
    }
    ap_uint<1> reducer3;
    reducer3 = (ap_uint<1>)0;
    ap_uint<1> conv1_pad[324];
    for (bit32 index_tuple = 0; index_tuple < 18; ++index_tuple) {
      for (bit32 i = 0; i < 18; ++i) {
        conv1_pad[(i + (index_tuple * 18))] = (((((1 <= index_tuple) && (index_tuple < 17)) && (1 <= i)) && (i < 17)) ? ((ap_uint<1>)input_image[((i + (index_tuple * 16)) + -17)]) : ((ap_uint<1>)0));
      }
    }
    ap_int<6> conv1_sum;
    conv1_sum = (ap_int<6>)0;
    ap_uint<1> bn1[4096];
    for (bit32 c = 0; c < 16; ++c) {
      for (bit32 h = 0; h < 16; ++h) {
        for (bit32 w = 0; w < 16; ++w) {
        #pragma HLS pipeline
          ap_int<6> conv1;
          conv1_sum = (ap_int<6>)0;
          for (bit32 ry = 0; ry < 3; ++ry) {
            for (bit32 rx = 0; rx < 3; ++rx) {
              conv1_sum = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)rx)) <= ((ap_int<33>)w)) && (((ap_int<33>)w) < ((ap_int<33>)17 - ((ap_int<33>)rx)))) && (((ap_int<33>)1 - ((ap_int<33>)ry)) <= ((ap_int<33>)h))) && (((ap_int<33>)h) < ((ap_int<33>)17 - ((ap_int<33>)ry)))) ? ((ubit32)((((ubit32)(conv1_pad[((w + rx) + ((h + ry) * 18))] == w_conv1[((rx + (ry * 3)) + (c * 9))])) * 2U) - 1U)) : ((ubit32)0U))) + ((ap_int<34>)conv1_sum)));
            }
          }
          conv1 = conv1_sum;
          bn1[((w + (h * 16)) + (c * 256))] = ((ap_uint<1>)((bn_t1[((w + (h * 16)) + (c * 256))] < ((ap_fixed<20, 10>)conv1)) ? 1 : 0));
        }
      }
    }
    ap_uint<1> reducer2;
    reducer2 = (ap_uint<1>)0;
    ap_uint<1> maxpool1[1024];
    for (bit32 i1 = 0; i1 < 1; ++i1) {
      for (bit32 c1 = 0; c1 < 16; ++c1) {
        for (bit32 h1 = 0; h1 < 8; ++h1) {
        #pragma HLS pipeline
          for (bit32 w1 = 0; w1 < 8; ++w1) {
            reducer2 = (ap_uint<1>)0;
            for (bit32 ra6 = 0; ra6 < 2; ++ra6) {
              for (bit32 ra7 = 0; ra7 < 2; ++ra7) {
                reducer2 = std::max(bn1[(((((w1 * 2) + ra7) + (((h1 * 2) + ra6) * 16)) + (c1 * 256)) + (i1 * 4096))], reducer2);
              }
            }
            maxpool1[(((w1 + (h1 * 8)) + (c1 * 64)) + (i1 * 1024))] = ((ap_uint<1>)((0U < ((ubit32)reducer2)) ? 1 : 0));
          }
        }
      }
    }
    ap_uint<1> conv2_pad[1600];
    for (bit32 not_zero = 0; not_zero < 16; ++not_zero) {
      for (bit32 index_tuple1 = 0; index_tuple1 < 10; ++index_tuple1) {
        for (bit32 i2 = 0; i2 < 10; ++i2) {
          conv2_pad[((i2 + (index_tuple1 * 10)) + (not_zero * 100))] = (((((1 <= index_tuple1) && (index_tuple1 < 9)) && (1 <= i2)) && (i2 < 9)) ? ((ap_uint<1>)maxpool1[(((i2 + (index_tuple1 * 8)) + (not_zero * 64)) + -9)]) : ((ap_uint<1>)0));
        }
      }
    }
    ap_int<6> conv2_sum;
    conv2_sum = (ap_int<6>)0;
    ap_uint<1> bn2[2048];
    for (bit32 c2 = 0; c2 < 32; ++c2) {
      for (bit32 h2 = 0; h2 < 8; ++h2) {
        for (bit32 w2 = 0; w2 < 8; ++w2) {
          ap_int<6> conv2;
          conv2_sum = (ap_int<6>)0;
          for (bit32 rc = 0; rc < 16; ++rc) {
          #pragma HLS pipeline
            for (bit32 ry1 = 0; ry1 < 3; ++ry1) {
              for (bit32 rx1 = 0; rx1 < 3; ++rx1) {
                conv2_sum = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)rx1)) <= ((ap_int<33>)w2)) && (((ap_int<33>)w2) < ((ap_int<33>)9 - ((ap_int<33>)rx1)))) && (((ap_int<33>)1 - ((ap_int<33>)ry1)) <= ((ap_int<33>)h2))) && (((ap_int<33>)h2) < ((ap_int<33>)9 - ((ap_int<33>)ry1)))) ? ((ubit32)((((ubit32)(conv2_pad[(((w2 + rx1) + ((h2 + ry1) * 10)) + (rc * 100))] == w_conv2[(((rx1 + (ry1 * 3)) + (rc * 9)) + (c2 * 144))])) * 2U) - 1U)) : ((ubit32)0U))) + ((ap_int<34>)conv2_sum)));
              }
            }
          }
          conv2 = conv2_sum;
          bn2[((w2 + (h2 * 8)) + (c2 * 64))] = ((ap_uint<1>)((bn_t2[((w2 + (h2 * 8)) + (c2 * 64))] < ((ap_fixed<20, 10>)conv2)) ? 1 : 0));
        }
      }
    }
    ap_uint<1> maxpool2[512];
    for (bit32 i3 = 0; i3 < 1; ++i3) {
      for (bit32 c3 = 0; c3 < 32; ++c3) {
        for (bit32 h3 = 0; h3 < 4; ++h3) {
        #pragma HLS pipeline
          for (bit32 w3 = 0; w3 < 4; ++w3) {
            reducer3 = (ap_uint<1>)0;
            for (bit32 ra8 = 0; ra8 < 2; ++ra8) {
              for (bit32 ra9 = 0; ra9 < 2; ++ra9) {
                reducer3 = std::max(bn2[(((((w3 * 2) + ra9) + (((h3 * 2) + ra8) * 8)) + (c3 * 64)) + (i3 * 2048))], reducer3);
              }
            }
            maxpool2[(((w3 + (h3 * 4)) + (c3 * 16)) + (i3 * 512))] = ((ap_uint<1>)((0U < ((ubit32)reducer3)) ? 1 : 0));
          }
        }
      }
    }
    ap_uint<1> flatten[512];
    for (bit32 i4 = 0; i4 < 1; ++i4) {
      for (bit32 j = 0; j < 512; ++j) {
      #pragma HLS pipeline
        flatten[(j + (i4 * 512))] = maxpool2[(((((j / 128) * 4) + ((j / 32) % 4)) + ((j % 32) * 16)) + (i4 * 512))];
      }
    }
    ap_fixed<20, 10> fc1_sum;
    fc1_sum = ((ap_fixed<20, 10>)0);
    ap_fixed<20, 10> fc1_matmul[256];
    for (bit32 i5 = 0; i5 < 1; ++i5) {
      for (bit32 j1 = 0; j1 < 256; ++j1) {
      #pragma HLS pipeline
        fc1_sum = ((ap_fixed<20, 10>)0);
        for (bit32 ra10 = 0; ra10 < 512; ++ra10) {
          fc1_sum = ((ap_fixed<20, 10>)(((ap_fixed<22, 12>)(flatten[(ra10 + (i5 * 512))] == w_fc1[(ra10 + (j1 * 512))])) + ((ap_fixed<22, 12>)fc1_sum)));
        }
        fc1_matmul[(j1 + (i5 * 256))] = ((ap_fixed<20, 10>)((((float)(((ap_fixed<53, 43>)(((ap_fixed<52, 42>)fc1_sum) * (ap_int<52>)2)) + (ap_fixed<53, 43>)-512)) * 6.250000e-02f) + ((float)b_fc1[j1])));
      }
    }
    ap_uint<1> fc1[256];
    for (bit32 i6 = 0; i6 < 1; ++i6) {
      for (bit32 j2 = 0; j2 < 256; ++j2) {
      #pragma HLS pipeline
        fc1[(j2 + (i6 * 256))] = ((ap_uint<1>)(((ap_fixed<42, 32>)0 < ((ap_fixed<42, 32>)fc1_matmul[(j2 + (i6 * 256))])) ? 1 : 0));
      }
    }
    ap_fixed<20, 10> fc2_sum;
    fc2_sum = ((ap_fixed<20, 10>)0);
    for (bit32 i7 = 0; i7 < 1; ++i7) {
      for (bit32 j3 = 0; j3 < 10; ++j3) {
      #pragma HLS pipeline
        fc2_sum = ((ap_fixed<20, 10>)0);
        for (bit32 ra11 = 0; ra11 < 256; ++ra11) {
          fc2_sum = ((ap_fixed<20, 10>)(((ap_fixed<22, 12>)(fc1[(ra11 + (i7 * 256))] == w_fc2[(ra11 + (j3 * 256))])) + ((ap_fixed<22, 12>)fc2_sum)));
        }
        fc2[(j3 + (i7 * 10))] = ((ap_fixed<20, 10>)((((float)(((ap_fixed<53, 43>)(((ap_fixed<52, 42>)fc2_sum) * (ap_int<52>)2)) + (ap_fixed<53, 43>)-256)) * 8.838835e-02f) + ((float)b_fc2[j3])));
      }
    }
    for (bit32 fc21 = 0; fc21 < 10; ++fc21) {
      fc2_channel.write(fc2[fc21]);
    }
  }

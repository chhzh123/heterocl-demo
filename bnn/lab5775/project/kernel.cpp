#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include "kernel.h"
  void test(hls::stream<ubit32 >& w_fc2_channel, hls::stream<ap_fixed<20, 10> >& b_fc1_channel, hls::stream<ubit32 >& w_fc1_channel, hls::stream<ap_fixed<20, 10> >& bn_t2_channel, hls::stream<ap_uint<16> >& w_conv2_channel, hls::stream<ap_fixed<20, 10> >& bn_t1_channel, hls::stream<ap_uint<1> >& w_conv1_channel, hls::stream<ap_uint<1> >& input_image_channel, hls::stream<ap_fixed<20, 10> >& b_fc2_channel, hls::stream<ap_fixed<20, 10> >& fc2_channel) {
  #pragma HLS INTERFACE axis port=w_fc2_channel offset=slave bundle=gmem0
  #pragma HLS INTERFACE axis port=b_fc1_channel offset=slave bundle=gmem1
  #pragma HLS INTERFACE axis port=w_fc1_channel offset=slave bundle=gmem2
  #pragma HLS INTERFACE axis port=bn_t2_channel offset=slave bundle=gmem3
  #pragma HLS INTERFACE axis port=w_conv2_channel offset=slave bundle=gmem4
  #pragma HLS INTERFACE axis port=bn_t1_channel offset=slave bundle=gmem5
  #pragma HLS INTERFACE axis port=w_conv1_channel offset=slave bundle=gmem6
  #pragma HLS INTERFACE axis port=input_image_channel offset=slave bundle=gmem7
  #pragma HLS INTERFACE axis port=b_fc2_channel offset=slave bundle=gmem8
  #pragma HLS INTERFACE axis port=fc2_channel offset=slave bundle=gmem9
  #pragma HLS INTERFACE s_axilite port=return bundle=control
    ap_fixed<20, 10> fc2[10];
    ubit32 w_fc2[80];
LOOP1: for (bit32 w_fc20 = 0; w_fc20 < 10; ++w_fc20) {
      for (bit32 w_fc21 = 0; w_fc21 < 8; ++w_fc21) {
        w_fc2[(w_fc21 + (w_fc20 * 8))] = w_fc2_channel.read();
      }
    }
    ap_fixed<20, 10> b_fc1[256];
LOOP2: for (bit32 b_fc10 = 0; b_fc10 < 256; ++b_fc10) {
      b_fc1[b_fc10] = b_fc1_channel.read();
    }
    ubit32 w_fc1[4096];
LOOP3: for (bit32 w_fc10 = 0; w_fc10 < 256; ++w_fc10) {
      for (bit32 w_fc11 = 0; w_fc11 < 16; ++w_fc11) {
        w_fc1[(w_fc11 + (w_fc10 * 16))] = w_fc1_channel.read();
      }
    }
    ap_fixed<20, 10> bn_t2[2048];
LOOP4: for (bit32 bn_t20 = 0; bn_t20 < 32; ++bn_t20) {
      for (bit32 bn_t21 = 0; bn_t21 < 8; ++bn_t21) {
        for (bit32 bn_t22 = 0; bn_t22 < 8; ++bn_t22) {
          bn_t2[((bn_t22 + (bn_t21 * 8)) + (bn_t20 * 64))] = bn_t2_channel.read();
        }
      }
    }
    ap_uint<16> w_conv2[288];
LOOP5: for (bit32 w_conv20 = 0; w_conv20 < 32; ++w_conv20) {
      for (bit32 w_conv22 = 0; w_conv22 < 3; ++w_conv22) {
        for (bit32 w_conv23 = 0; w_conv23 < 3; ++w_conv23) {
          w_conv2[((w_conv23 + (w_conv22 * 3)) + (w_conv20 * 9))] = w_conv2_channel.read();
        }
      }
    }
    ap_fixed<20, 10> bn_t1[4096];
LOOP6: for (bit32 bn_t10 = 0; bn_t10 < 16; ++bn_t10) {
      for (bit32 bn_t11 = 0; bn_t11 < 16; ++bn_t11) {
        for (bit32 bn_t12 = 0; bn_t12 < 16; ++bn_t12) {
          bn_t1[((bn_t12 + (bn_t11 * 16)) + (bn_t10 * 256))] = bn_t1_channel.read();
        }
      }
    }
    ap_uint<1> w_conv1[144];
LOOP7: for (bit32 w_conv10 = 0; w_conv10 < 16; ++w_conv10) {
      for (bit32 w_conv12 = 0; w_conv12 < 3; ++w_conv12) {
        for (bit32 w_conv13 = 0; w_conv13 < 3; ++w_conv13) {
          w_conv1[((w_conv13 + (w_conv12 * 3)) + (w_conv10 * 9))] = w_conv1_channel.read();
        }
      }
    }
    ap_uint<1> input_image[256];
LOOP8: for (bit32 input_image2 = 0; input_image2 < 16; ++input_image2) {
      for (bit32 input_image3 = 0; input_image3 < 16; ++input_image3) {
        input_image[(input_image3 + (input_image2 * 16))] = input_image_channel.read();
      }
    }
    ap_fixed<20, 10> b_fc2[10];
LOOP9: for (bit32 b_fc20 = 0; b_fc20 < 10; ++b_fc20) {
      b_fc2[b_fc20] = b_fc2_channel.read();
    }
    bit32 fc2_popcnt;
    fc2_popcnt = 0;
    bit32 fc1_popcnt;
    fc1_popcnt = 0;
    ap_int<6> conv2_sum;
    conv2_sum = (ap_int<6>)0;
    ap_uint<1> conv1_pad[324];
LOOP10: for (bit32 index_tuple = 0; index_tuple < 18; ++index_tuple) {
    #pragma HLS pipeline
      for (bit32 i = 0; i < 18; ++i) {
        conv1_pad[(i + (index_tuple * 18))] = (((((1 <= index_tuple) && (index_tuple < 17)) && (1 <= i)) && (i < 17)) ? ((ap_uint<1>)input_image[((i + (index_tuple * 16)) + -17)]) : ((ap_uint<1>)0));
      }
    }
    ap_int<6> conv1_sum;
    conv1_sum = (ap_int<6>)0;
    ap_int<6> conv1[4096];
LOOP11: for (bit32 ff = 0; ff < 16; ++ff) {
      for (bit32 yy = 0; yy < 16; ++yy) {
        for (bit32 xx = 0; xx < 16; ++xx) {
        #pragma HLS pipeline
          conv1_sum = (ap_int<6>)0;
          for (bit32 ry = 0; ry < 3; ++ry) {
            for (bit32 rx = 0; rx < 3; ++rx) {
              conv1_sum = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)rx)) <= ((ap_int<33>)xx)) && (((ap_int<33>)xx) < ((ap_int<33>)17 - ((ap_int<33>)rx)))) && (((ap_int<33>)1 - ((ap_int<33>)ry)) <= ((ap_int<33>)yy))) && (((ap_int<33>)yy) < ((ap_int<33>)17 - ((ap_int<33>)ry)))) ? ((ubit32)((((1U - ((ubit32)conv1_pad[((xx + rx) + ((yy + ry) * 18))])) ^ w_conv1[((rx + (ry * 3)) + (ff * 9))]) << 1) - 1U)) : ((ubit32)0U))) + ((ap_int<34>)conv1_sum)));
            }
          }
          conv1[((xx + (yy * 16)) + (ff * 256))] = conv1_sum;
        }
      }
    }
    ap_uint<16> bn1_pack;
    bn1_pack = (ap_uint<16>)0;
    ap_uint<16> bn1[256];
LOOP12: for (bit32 h = 0; h < 16; ++h) {
      for (bit32 w = 0; w < 16; ++w) {
      #pragma HLS pipeline
        bn1_pack = (ap_uint<16>)0;
        for (bit32 i1 = 0; i1 < 16; ++i1) {
          bn1_pack(i1, i1) = ((bn_t1[((w + (h * 16)) + (i1 * 256))] < ((ap_fixed<20, 10>)conv1[((w + (h * 16)) + (i1 * 256))])) ? ((bit32)1) : ((bit32)0));
        }
        bn1[(w + (h * 16))] = bn1_pack;
      }
    }
    ap_uint<16> maxpool1[64];
LOOP13: for (bit32 i2 = 0; i2 < 1; ++i2) {
      for (bit32 h1 = 0; h1 < 8; ++h1) {
      #pragma HLS pipeline
        for (bit32 w1 = 0; w1 < 8; ++w1) {
          maxpool1[((w1 + (h1 * 8)) + (i2 * 64))] = (((bn1[(((w1 + (h1 * 16)) + (i2 * 128)) * 2)] | bn1[((((w1 + (h1 * 16)) + (i2 * 128)) * 2) + 1)]) | bn1[((((w1 + (h1 * 16)) + (i2 * 128)) * 2) + 16)]) | bn1[((((w1 + (h1 * 16)) + (i2 * 128)) * 2) + 17)]);
        }
      }
    }
    ap_uint<16> conv2_pad[100];
LOOP14: for (bit32 not_zero = 0; not_zero < 1; ++not_zero) {
    #pragma HLS pipeline
      for (bit32 index_tuple1 = 0; index_tuple1 < 10; ++index_tuple1) {
        for (bit32 i3 = 0; i3 < 10; ++i3) {
          conv2_pad[((i3 + (index_tuple1 * 10)) + (not_zero * 100))] = (((((1 <= index_tuple1) && (index_tuple1 < 9)) && (1 <= i3)) && (i3 < 9)) ? ((ap_uint<16>)maxpool1[(((i3 + (index_tuple1 * 8)) + (not_zero * 64)) + -9)]) : ((ap_uint<16>)0));
        }
      }
    }
    ap_int<6> conv2[2048];
LOOP15: for (bit32 ff1 = 0; ff1 < 32; ++ff1) {
      for (bit32 yy1 = 0; yy1 < 8; ++yy1) {
        for (bit32 xx1 = 0; xx1 < 8; ++xx1) {
        #pragma HLS pipeline
          conv2_sum = (ap_int<6>)0;
          for (bit32 conv2_ry = 0; conv2_ry < 3; ++conv2_ry) {
            for (bit32 conv2_rx = 0; conv2_rx < 3; ++conv2_rx) {
              for (bit32 conv2_rb = 0; conv2_rb < 16; ++conv2_rb) {
                conv2_sum = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)conv2_rx)) <= ((ap_int<33>)xx1)) && (((ap_int<33>)xx1) < ((ap_int<33>)9 - ((ap_int<33>)conv2_rx)))) && (((ap_int<33>)1 - ((ap_int<33>)conv2_ry)) <= ((ap_int<33>)yy1))) && (((ap_int<33>)yy1) < ((ap_int<33>)9 - ((ap_int<33>)conv2_ry)))) ? ((ubit32)(((65535U - ((ubit32)(conv2_pad[((xx1 + conv2_rx) + ((yy1 + conv2_ry) * 10))] ^ w_conv2[((conv2_rx + (conv2_ry * 3)) + (ff1 * 9))])))[conv2_rb] << 1) - 1U)) : ((ubit32)0U))) + ((ap_int<34>)conv2_sum)));
              }
            }
          }
          conv2[((xx1 + (yy1 * 8)) + (ff1 * 64))] = conv2_sum;
        }
      }
    }
    ubit32 bn2_pack;
    bn2_pack = 0U;
    ubit32 bn2[64];
LOOP16: for (bit32 h2 = 0; h2 < 8; ++h2) {
      for (bit32 w2 = 0; w2 < 8; ++w2) {
      #pragma HLS pipeline
        bn2_pack = 0U;
        for (bit32 i4 = 0; i4 < 32; ++i4) {
          bn2_pack(i4, i4) = ((bn_t2[((w2 + (h2 * 8)) + (i4 * 64))] < ((ap_fixed<20, 10>)conv2[((w2 + (h2 * 8)) + (i4 * 64))])) ? ((bit32)1) : ((bit32)0));
        }
        bn2[(w2 + (h2 * 8))] = bn2_pack;
      }
    }
    ubit32 maxpool2[16];
LOOP17: for (bit32 i5 = 0; i5 < 1; ++i5) {
      for (bit32 h3 = 0; h3 < 4; ++h3) {
      #pragma HLS pipeline
        for (bit32 w3 = 0; w3 < 4; ++w3) {
          maxpool2[((w3 + (h3 * 4)) + (i5 * 16))] = (((bn2[(((w3 + (h3 * 8)) + (i5 * 32)) * 2)] | bn2[((((w3 + (h3 * 8)) + (i5 * 32)) * 2) + 1)]) | bn2[((((w3 + (h3 * 8)) + (i5 * 32)) * 2) + 8)]) | bn2[((((w3 + (h3 * 8)) + (i5 * 32)) * 2) + 9)]);
        }
      }
    }
    bit32 packed_flatten[16];
LOOP18: for (bit32 i6 = 0; i6 < 1; ++i6) {
      for (bit32 j = 0; j < 16; ++j) {
      #pragma HLS pipeline
        packed_flatten[(j + (i6 * 16))] = ((bit32)maxpool2[(j + (i6 * 16))]);
      }
    }
    bit32 fc1_xor[4096];
LOOP19: for (bit32 i7 = 0; i7 < 1; ++i7) {
      for (bit32 j1 = 0; j1 < 256; ++j1) {
      #pragma HLS pipeline
        for (bit32 u = 0; u < 16; ++u) {
          fc1_xor[((u + (j1 * 16)) + (i7 * 4096))] = (packed_flatten[(u + (i7 * 16))] ^ w_fc1[(u + (j1 * 16))]);
        }
      }
    }
    bit32 fc1_matmul[256];
LOOP20: for (bit32 j2 = 0; j2 < 256; ++j2) {
      fc1_popcnt = 0;
      for (bit32 fc1_rk = 0; fc1_rk < 16; ++fc1_rk) {
      #pragma HLS pipeline
        for (bit32 fc1_rb = 0; fc1_rb < 32; ++fc1_rb) {
          fc1_popcnt = ((bit32)(((ap_int<33>)fc1_xor[(fc1_rk + (j2 * 16))][fc1_rb]) + ((ap_int<33>)fc1_popcnt)));
        }
      }
      fc1_matmul[j2] = fc1_popcnt;
    }
    ap_fixed<20, 10> fc1_bias[256];
LOOP21: for (bit32 i8 = 0; i8 < 1; ++i8) {
      for (bit32 j3 = 0; j3 < 256; ++j3) {
      #pragma HLS pipeline
        fc1_bias[(j3 + (i8 * 256))] = ((ap_fixed<20, 10>)((((float)(512 - (fc1_matmul[(j3 + (i8 * 256))] << 1))) * 3.535534e-01f) + ((float)b_fc1[j3])));
      }
    }
    bit32 fc1_pack;
    fc1_pack = 0;
    bit32 fc1[8];
LOOP22: for (bit32 i9 = 0; i9 < 1; ++i9) {
      for (bit32 j4 = 0; j4 < 8; ++j4) {
      #pragma HLS pipeline
        fc1_pack = 0;
        for (bit32 i10 = 0; i10 < 32; ++i10) {
          fc1_pack(i10, i10) = (((ap_fixed<42, 32>)0 < ((ap_fixed<42, 32>)fc1_bias[(((j4 * 32) + i10) + (i9 * 256))])) ? ((bit32)1) : ((bit32)0));
        }
        fc1[(j4 + (i9 * 8))] = fc1_pack;
      }
    }
    bit32 fc2_xor[80];
LOOP23: for (bit32 i11 = 0; i11 < 1; ++i11) {
      for (bit32 j5 = 0; j5 < 10; ++j5) {
      #pragma HLS pipeline
        for (bit32 u1 = 0; u1 < 8; ++u1) {
          fc2_xor[((u1 + (j5 * 8)) + (i11 * 80))] = (fc1[(u1 + (i11 * 8))] ^ w_fc2[(u1 + (j5 * 8))]);
        }
      }
    }
    bit32 fc2_matmul[10];
LOOP24: for (bit32 j6 = 0; j6 < 10; ++j6) {
    #pragma HLS pipeline
      fc2_popcnt = 0;
      for (bit32 fc2_rk = 0; fc2_rk < 8; ++fc2_rk) {
        for (bit32 fc2_rb = 0; fc2_rb < 32; ++fc2_rb) {
          fc2_popcnt = ((bit32)(((ap_int<33>)fc2_xor[(fc2_rk + (j6 * 8))][fc2_rb]) + ((ap_int<33>)fc2_popcnt)));
        }
      }
      fc2_matmul[j6] = fc2_popcnt;
    }
LOOP25: for (bit32 i12 = 0; i12 < 1; ++i12) {
      for (bit32 j7 = 0; j7 < 10; ++j7) {
      #pragma HLS pipeline
        fc2[(j7 + (i12 * 10))] = ((ap_fixed<20, 10>)((((float)(256 - (fc2_matmul[(j7 + (i12 * 10))] << 1))) * 5.000000e-01f) + ((float)b_fc2[j7])));
      }
    }
LOOP26: for (bit32 fc21 = 0; fc21 < 10; ++fc21) {
      fc2_channel.write(fc2[fc21]);
    }
  }

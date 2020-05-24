#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include "kernel.h"
  void test(hls::stream<ap_fixed<20, 10> >& b_fc2_channel, hls::stream<ubit32 >& w_fc2_channel, hls::stream<ubit32 >& w_fc1_channel, hls::stream<ap_fixed<20, 10> >& bn_t2_channel, hls::stream<ap_uint<1> >& w_conv1_channel, hls::stream<ap_uint<1> >& input_image_channel, hls::stream<ap_fixed<20, 10> >& bn_t1_channel, hls::stream<ap_uint<1> >& w_conv2_channel, hls::stream<ap_fixed<20, 10> >& b_fc1_channel, hls::stream<ap_fixed<20, 10> >& fc2_channel) {
  #pragma HLS INTERFACE axis port=b_fc2_channel offset=slave bundle=gmem0
  #pragma HLS INTERFACE axis port=w_fc2_channel offset=slave bundle=gmem1
  #pragma HLS INTERFACE axis port=w_fc1_channel offset=slave bundle=gmem2
  #pragma HLS INTERFACE axis port=bn_t2_channel offset=slave bundle=gmem3
  #pragma HLS INTERFACE axis port=w_conv1_channel offset=slave bundle=gmem4
  #pragma HLS INTERFACE axis port=input_image_channel offset=slave bundle=gmem5
  #pragma HLS INTERFACE axis port=bn_t1_channel offset=slave bundle=gmem6
  #pragma HLS INTERFACE axis port=w_conv2_channel offset=slave bundle=gmem7
  #pragma HLS INTERFACE axis port=b_fc1_channel offset=slave bundle=gmem8
  #pragma HLS INTERFACE axis port=fc2_channel offset=slave bundle=gmem9
  #pragma HLS INTERFACE s_axilite port=return bundle=control
    ap_fixed<20, 10> fc2[10];
    ap_fixed<20, 10> b_fc2[10];
    for (bit32 b_fc20 = 0; b_fc20 < 10; ++b_fc20) {
      b_fc2[b_fc20] = b_fc2_channel.read();
    }
    ubit32 w_fc2[80];
    for (bit32 w_fc20 = 0; w_fc20 < 10; ++w_fc20) {
      for (bit32 w_fc21 = 0; w_fc21 < 8; ++w_fc21) {
        w_fc2[(w_fc21 + (w_fc20 * 8))] = w_fc2_channel.read();
      }
    }
    ubit32 w_fc1[4096];
    for (bit32 w_fc10 = 0; w_fc10 < 256; ++w_fc10) {
      for (bit32 w_fc11 = 0; w_fc11 < 16; ++w_fc11) {
        w_fc1[(w_fc11 + (w_fc10 * 16))] = w_fc1_channel.read();
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
    ap_fixed<20, 10> bn_t1[4096];
    for (bit32 bn_t10 = 0; bn_t10 < 16; ++bn_t10) {
      for (bit32 bn_t11 = 0; bn_t11 < 16; ++bn_t11) {
        for (bit32 bn_t12 = 0; bn_t12 < 16; ++bn_t12) {
          bn_t1[((bn_t12 + (bn_t11 * 16)) + (bn_t10 * 256))] = bn_t1_channel.read();
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
    ap_fixed<20, 10> b_fc1[256];
    for (bit32 b_fc10 = 0; b_fc10 < 256; ++b_fc10) {
      b_fc1[b_fc10] = b_fc1_channel.read();
    }
    bit32 fc2_popcnt;
    fc2_popcnt = 0;
    bit32 fc1_popcnt;
    fc1_popcnt = 0;
    ubit32 pack_temp;
    pack_temp = 0U;
    ubit32 packed_bn2_temp;
    packed_bn2_temp = 0U;
    ap_int<6> conv2_sum;
    conv2_sum = (ap_int<6>)0;
    ap_uint<16> packed_bn1_temp;
    packed_bn1_temp = (ap_uint<16>)0;
    ap_uint<1> conv1_pad[324];
    for (bit32 not_zero = 0; not_zero < 1; ++not_zero) {
    #pragma HLS pipeline
      for (bit32 index_tuple = 0; index_tuple < 18; ++index_tuple) {
        for (bit32 i = 0; i < 18; ++i) {
          conv1_pad[((i + (index_tuple * 18)) + (not_zero * 324))] = (((((1 <= index_tuple) && (index_tuple < 17)) && (1 <= i)) && (i < 17)) ? ((ap_uint<1>)input_image[(((i + (index_tuple * 16)) + (not_zero * 256)) + -17)]) : ((ap_uint<1>)0));
        }
      }
    }
    ap_int<6> conv1_sum;
    conv1_sum = (ap_int<6>)0;
    ap_uint<1> bn1[4096];
    for (bit32 c = 0; c < 16; ++c) {
      for (bit32 h = 0; h < 16; ++h) {
      #pragma HLS pipeline
        for (bit32 w = 0; w < 16; ++w) {
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
    ap_uint<16> packed_bn1[256];
    for (bit32 i1 = 0; i1 < 16; ++i1) {
      for (bit32 new_indices = 0; new_indices < 16; ++new_indices) {
      #pragma HLS pipeline
        packed_bn1_temp = (ap_uint<16>)0;
        for (bit32 i2 = 0; i2 < 16; ++i2) {
          packed_bn1_temp(i2, i2) = bn1[((new_indices + (i1 * 16)) + (i2 * 256))];
        }
        packed_bn1[(new_indices + (i1 * 16))] = packed_bn1_temp;
      }
    }
    ap_uint<1> maxpool1[2048];
    for (bit32 i3 = 0; i3 < 1; ++i3) {
      for (bit32 c1 = 0; c1 < 32; ++c1) {
      #pragma HLS pipeline
        for (bit32 h1 = 0; h1 < 8; ++h1) {
          for (bit32 w1 = 0; w1 < 8; ++w1) {
            maxpool1[(((w1 + (h1 * 8)) + (c1 * 64)) + (i3 * 2048))] = ((ap_uint<1>)(((packed_bn1[(((w1 + (h1 * 16)) + (i3 * 128)) * 2)][c1] | packed_bn1[((((w1 + (h1 * 16)) + (i3 * 128)) * 2) + 1)][c1]) | packed_bn1[((((w1 + (h1 * 16)) + (i3 * 128)) * 2) + 16)][c1]) | packed_bn1[((((w1 + (h1 * 16)) + (i3 * 128)) * 2) + 17)][c1]));
          }
        }
      }
    }
    ap_uint<1> conv2_pad[3200];
    for (bit32 not_zero1 = 0; not_zero1 < 32; ++not_zero1) {
    #pragma HLS pipeline
      for (bit32 index_tuple1 = 0; index_tuple1 < 10; ++index_tuple1) {
        for (bit32 i4 = 0; i4 < 10; ++i4) {
          conv2_pad[((i4 + (index_tuple1 * 10)) + (not_zero1 * 100))] = (((((1 <= index_tuple1) && (index_tuple1 < 9)) && (1 <= i4)) && (i4 < 9)) ? ((ap_uint<1>)maxpool1[(((i4 + (index_tuple1 * 8)) + (not_zero1 * 64)) + -9)]) : ((ap_uint<1>)0));
        }
      }
    }
    ap_uint<1> bn2[2048];
    for (bit32 c2 = 0; c2 < 32; ++c2) {
      for (bit32 h2 = 0; h2 < 8; ++h2) {
        for (bit32 w2 = 0; w2 < 8; ++w2) {
        #pragma HLS pipeline
          ap_int<6> conv2;
          conv2_sum = (ap_int<6>)0;
          for (bit32 rc = 0; rc < 16; ++rc) {
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
    ubit32 packed_bn2[64];
    for (bit32 i5 = 0; i5 < 8; ++i5) {
      for (bit32 new_indices1 = 0; new_indices1 < 8; ++new_indices1) {
      #pragma HLS pipeline
        packed_bn2_temp = 0U;
        for (bit32 i6 = 0; i6 < 32; ++i6) {
          packed_bn2_temp(i6, i6) = bn2[((new_indices1 + (i5 * 8)) + (i6 * 64))];
        }
        packed_bn2[(new_indices1 + (i5 * 8))] = packed_bn2_temp;
      }
    }
    ap_uint<1> maxpool2[512];
    for (bit32 i7 = 0; i7 < 1; ++i7) {
      for (bit32 c3 = 0; c3 < 32; ++c3) {
      #pragma HLS pipeline
        for (bit32 h3 = 0; h3 < 4; ++h3) {
          for (bit32 w3 = 0; w3 < 4; ++w3) {
            maxpool2[(((w3 + (h3 * 4)) + (c3 * 16)) + (i7 * 512))] = ((ap_uint<1>)(((packed_bn2[(((w3 + (h3 * 8)) + (i7 * 32)) * 2)][c3] | packed_bn2[((((w3 + (h3 * 8)) + (i7 * 32)) * 2) + 1)][c3]) | packed_bn2[((((w3 + (h3 * 8)) + (i7 * 32)) * 2) + 8)][c3]) | packed_bn2[((((w3 + (h3 * 8)) + (i7 * 32)) * 2) + 9)][c3]));
          }
        }
      }
    }
    ap_uint<1> flatten[512];
    for (bit32 i8 = 0; i8 < 1; ++i8) {
      for (bit32 j = 0; j < 512; ++j) {
      #pragma HLS pipeline
        flatten[(j + (i8 * 512))] = maxpool2[(((((j / 128) * 4) + ((j / 32) % 4)) + ((j % 32) * 16)) + (i8 * 512))];
      }
    }
    ubit32 pack[16];
    for (bit32 temp = 0; temp < 16; ++temp) {
    #pragma HLS pipeline
      pack_temp = 0U;
      for (bit32 i9 = 0; i9 < 32; ++i9) {
        pack_temp(i9, i9) = flatten[((temp * 32) + i9)];
      }
      pack[temp] = pack_temp;
    }
    float reducer2;
    reducer2 = 0.000000e+00f;
    ap_uint<1> fc1[256];
    for (bit32 i10 = 0; i10 < 1; ++i10) {
      for (bit32 j1 = 0; j1 < 256; ++j1) {
      #pragma HLS pipeline
        ap_fixed<20, 10> fc1_bias;
        ubit32 fc1_matmul;
        reducer2 = 0.000000e+00f;
        for (bit32 ra10 = 0; ra10 < 16; ++ra10) {
          ubit32 fc1_popcount;
          bit32 fc1_xor;
          fc1_xor = ((bit32)(pack[(ra10 + (i10 * 16))] ^ w_fc1[(ra10 + (j1 * 16))]));
          fc1_popcnt = 0;
          for (bit32 i11 = 0; i11 < 32; ++i11) {
            fc1_popcnt = ((bit32)(((ap_int<33>)fc1_popcnt) + ((ap_int<33>)fc1_xor[i11])));
          }
          fc1_popcount = ((ubit32)fc1_popcnt);
          reducer2 = (((float)fc1_popcount) + reducer2);
        }
        fc1_matmul = ((ubit32)(5.120000e+02f - (reducer2 * 2.000000e+00f)));
        fc1_bias = ((ap_fixed<20, 10>)((((float)fc1_matmul) * 3.535534e-01f) + ((float)b_fc1[j1])));
        fc1[(j1 + (i10 * 256))] = ((ap_uint<1>)(((ap_fixed<42, 32>)0 < ((ap_fixed<42, 32>)fc1_bias)) ? 1 : 0));
      }
    }
    ubit32 pack2_temp;
    pack2_temp = 0U;
    ubit32 pack2[8];
    for (bit32 temp1 = 0; temp1 < 8; ++temp1) {
    #pragma HLS pipeline
      pack2_temp = 0U;
      for (bit32 i12 = 0; i12 < 32; ++i12) {
        pack2_temp(i12, i12) = fc1[((temp1 * 32) + i12)];
      }
      pack2[temp1] = pack2_temp;
    }
    float reducer3;
    reducer3 = 0.000000e+00f;
    for (bit32 i13 = 0; i13 < 1; ++i13) {
      for (bit32 j2 = 0; j2 < 10; ++j2) {
      #pragma HLS pipeline
        ubit32 fc2_matmul;
        reducer3 = 0.000000e+00f;
        for (bit32 ra11 = 0; ra11 < 8; ++ra11) {
          ubit32 fc2_popcount;
          bit32 fc2_xor;
          fc2_xor = ((bit32)(pack2[(ra11 + (i13 * 8))] ^ w_fc2[(ra11 + (j2 * 8))]));
          fc2_popcnt = 0;
          for (bit32 i14 = 0; i14 < 32; ++i14) {
            fc2_popcnt = ((bit32)(((ap_int<33>)fc2_popcnt) + ((ap_int<33>)fc2_xor[i14])));
          }
          fc2_popcount = ((ubit32)fc2_popcnt);
          reducer3 = (((float)fc2_popcount) + reducer3);
        }
        fc2_matmul = ((ubit32)(2.560000e+02f - (reducer3 * 2.000000e+00f)));
        fc2[(j2 + (i13 * 10))] = ((ap_fixed<20, 10>)((((float)fc2_matmul) * 5.000000e-01f) + ((float)b_fc2[j2])));
      }
    }
    for (bit32 fc21 = 0; fc21 < 10; ++fc21) {
      fc2_channel.write(fc2[fc21]);
    }
  }

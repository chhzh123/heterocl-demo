#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
void default_function(ap_uint<1> input_image[1][1][16][16], ap_uint<1> w_conv1[16][1][3][3], ap_fixed<20, 10> bn_t1[16][16][16], ap_uint<16> w_conv2[32][1][3][3], ap_fixed<20, 10> bn_t2[32][8][8], ap_uint<32> w_fc1[256][16], ap_fixed<20, 10> b_fc1[256], ap_uint<32> w_fc2[10][8], ap_fixed<20, 10> b_fc2[10], ap_fixed<20, 10> fc2[1][10]) {
  #pragma HLS array_partition variable=bn_t2 complete dim=1
  #pragma HLS array_partition variable=w_conv2 complete dim=0
  #pragma HLS array_partition variable=bn_t1 complete dim=1
  #pragma HLS array_partition variable=w_conv1 complete dim=0
  #pragma HLS array_partition variable=input_image complete dim=0
  ap_int<32> _top;
  ap_uint<1> conv1_pad[1][1][18][18];
  #pragma HLS array_partition variable=conv1_pad complete dim=4
  ap_int<32> conv1_pad_partitioned;
LOOP1: for (ap_int<32> index_tuple = 0; index_tuple < 18; ++index_tuple) {
  #pragma HLS pipeline
    for (ap_int<32> i = 0; i < 18; ++i) {
      conv1_pad[0][0][index_tuple][i] = (((((1 <= index_tuple) && (index_tuple < 17)) && (1 <= i)) && (i < 17)) ? ((ap_uint<1>)input_image[((((i - ((i + -1) % 16)) + (index_tuple * 16)) + -17) / 256)][0][(((((i - ((i + -1) % 16)) + (index_tuple * 16)) + -17) / 16) % 16)][((i + -1) % 16)]) : ((ap_uint<1>)0));
    }
  }
  ap_int<6> conv1[1][16][16][16];
  #pragma HLS array_partition variable=conv1 complete dim=2
  ap_uint<1> LB[1][1][3][18];
  #pragma HLS array_partition variable=LB complete dim=3
  ap_uint<1> WB[1][1][3][3];
  #pragma HLS array_partition variable=WB complete dim=0
  ap_int<32> conv1_partitioned;
LOOP2: for (ap_int<32> ff = 0; ff < 16; ++ff) {
    for (ap_int<32> yy_reuse = 0; yy_reuse < 18; ++yy_reuse) {
      for (ap_int<32> xx_reuse = 0; xx_reuse < 18; ++xx_reuse) {
      #pragma HLS pipeline
        for (ap_int<32> conv1_pad_1 = 0; conv1_pad_1 < 2; ++conv1_pad_1) {
          LB[0][0][conv1_pad_1][xx_reuse] = LB[0][0][(conv1_pad_1 + 1)][xx_reuse];
        }
        LB[0][0][2][xx_reuse] = conv1_pad[0][0][yy_reuse][xx_reuse];
        if (2 <= yy_reuse) {
          for (ap_int<32> LB_1 = 0; LB_1 < 3; ++LB_1) {
            for (ap_int<32> LB_0 = 0; LB_0 < 2; ++LB_0) {
              WB[0][0][LB_1][LB_0] = WB[0][0][LB_1][(LB_0 + 1)];
            }
            WB[0][0][LB_1][2] = LB[0][0][LB_1][xx_reuse];
          }
          if (2 <= xx_reuse) {
            ap_int<6> conv1_sum;
            conv1_sum = (ap_int<6>)0;
            for (ap_int<32> conv1_ry = 0; conv1_ry < 3; ++conv1_ry) {
              for (ap_int<32> conv1_rx = 0; conv1_rx < 3; ++conv1_rx) {
                conv1_sum = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)conv1_rx)) <= ((ap_int<33>)(xx_reuse + -2))) && (((ap_int<33>)(xx_reuse + -2)) < ((ap_int<33>)17 - ((ap_int<33>)conv1_rx)))) && (((ap_int<33>)1 - ((ap_int<33>)conv1_ry)) <= ((ap_int<33>)(yy_reuse + -2)))) && (((ap_int<33>)(yy_reuse + -2)) < ((ap_int<33>)17 - ((ap_int<33>)conv1_ry)))) ? ((ap_uint<32>)(((1U - ((ap_uint<32>)(WB[0][0][conv1_ry][conv1_rx] ^ w_conv1[ff][0][conv1_ry][conv1_rx])))[0] << 1) - 1U)) : ((ap_uint<32>)0U))) + ((ap_int<34>)conv1_sum)));
              }
            }
            conv1[0][ff][(yy_reuse + -2)][(xx_reuse + -2)] = conv1_sum;
          }
        }
      }
    }
  }
  ap_uint<16> bn1[1][1][16][16];
  #pragma HLS array_partition variable=bn1 complete dim=4
  ap_int<32> bn1_partitioned;
LOOP3: for (ap_int<32> h = 0; h < 16; ++h) {
    for (ap_int<32> w = 0; w < 16; ++w) {
    #pragma HLS pipeline
      ap_uint<16> bn1_pack;
      bn1_pack = (ap_uint<16>)0;
      for (ap_int<32> i1 = 0; i1 < 16; ++i1) {
        bn1_pack(i1, i1) = ((bn_t1[i1][h][w] < ((ap_fixed<20, 10>)conv1[0][i1][h][w])) ? ((ap_int<32>)1) : ((ap_int<32>)0));
      }
      bn1[0][0][h][w] = bn1_pack;
    }
  }
  ap_uint<16> maxpool1[1][1][8][8];
  #pragma HLS array_partition variable=maxpool1 complete dim=4
  ap_int<32> maxpool1_partitioned;
LOOP4: for (ap_int<32> i2 = 0; i2 < 1; ++i2) {
    for (ap_int<32> h1 = 0; h1 < 8; ++h1) {
    #pragma HLS pipeline
      for (ap_int<32> w1 = 0; w1 < 8; ++w1) {
        maxpool1[i2][0][h1][w1] = (((bn1[i2][0][((w1 / 8) + (h1 * 2))][((((w1 + (h1 * 16)) + (i2 * 128)) * 2) % 16)] | bn1[0][0][(((((w1 + (h1 * 16)) + (i2 * 128)) * 2) + 1) / 16)][(((((w1 + (h1 * 16)) + (i2 * 128)) * 2) + 1) % 16)]) | bn1[i2][0][(((w1 / 8) + (h1 * 2)) + 1)][((((w1 + (h1 * 16)) + (i2 * 128)) * 2) % 16)]) | bn1[0][0][((((((w1 + (h1 * 16)) + (i2 * 128)) * 2) - (((((w1 + (h1 * 16)) + (i2 * 128)) * 2) + 1) % 16)) + 17) / 16)][(((((w1 + (h1 * 16)) + (i2 * 128)) * 2) + 1) % 16)]);
      }
    }
  }
  ap_uint<16> conv2_pad[1][1][10][10];
  #pragma HLS array_partition variable=conv2_pad complete dim=4
  ap_int<32> conv2_pad_partitioned;
LOOP5: for (ap_int<32> index_tuple1 = 0; index_tuple1 < 10; ++index_tuple1) {
  #pragma HLS pipeline
    for (ap_int<32> i3 = 0; i3 < 10; ++i3) {
      conv2_pad[0][0][index_tuple1][i3] = (((((1 <= index_tuple1) && (index_tuple1 < 9)) && (1 <= i3)) && (i3 < 9)) ? ((ap_uint<16>)maxpool1[((((i3 - ((i3 + -1) % 8)) + (index_tuple1 * 8)) + -9) / 64)][0][(((((i3 - ((i3 + -1) % 8)) + (index_tuple1 * 8)) + -9) / 8) % 8)][((i3 + -1) % 8)]) : ((ap_uint<16>)0));
    }
  }
  ap_int<6> conv2[1][32][8][8];
  #pragma HLS array_partition variable=conv2 complete dim=2
  ap_uint<16> LB1[1][1][3][10];
  #pragma HLS array_partition variable=LB1 complete dim=3
  ap_uint<16> WB1[1][1][3][3];
  #pragma HLS array_partition variable=WB1 complete dim=0
  ap_int<32> conv2_partitioned;
LOOP6: for (ap_int<32> ff1 = 0; ff1 < 32; ++ff1) {
    for (ap_int<32> yy_reuse1 = 0; yy_reuse1 < 10; ++yy_reuse1) {
      for (ap_int<32> xx_reuse1 = 0; xx_reuse1 < 10; ++xx_reuse1) {
      #pragma HLS pipeline
        for (ap_int<32> conv2_pad_1 = 0; conv2_pad_1 < 2; ++conv2_pad_1) {
          LB1[0][0][conv2_pad_1][xx_reuse1] = LB1[0][0][(conv2_pad_1 + 1)][xx_reuse1];
        }
        LB1[0][0][2][xx_reuse1] = conv2_pad[0][0][yy_reuse1][xx_reuse1];
        if (2 <= yy_reuse1) {
          for (ap_int<32> LB_11 = 0; LB_11 < 3; ++LB_11) {
            for (ap_int<32> LB_01 = 0; LB_01 < 2; ++LB_01) {
              WB1[0][0][LB_11][LB_01] = WB1[0][0][LB_11][(LB_01 + 1)];
            }
            WB1[0][0][LB_11][2] = LB1[0][0][LB_11][xx_reuse1];
          }
          if (2 <= xx_reuse1) {
            ap_int<6> conv2_sum;
            conv2_sum = (ap_int<6>)0;
            for (ap_int<32> conv2_ry = 0; conv2_ry < 3; ++conv2_ry) {
              for (ap_int<32> conv2_rx = 0; conv2_rx < 3; ++conv2_rx) {
                for (ap_int<32> conv2_rb = 0; conv2_rb < 16; ++conv2_rb) {
                  conv2_sum = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)conv2_rx)) <= ((ap_int<33>)(xx_reuse1 + -2))) && (((ap_int<33>)(xx_reuse1 + -2)) < ((ap_int<33>)9 - ((ap_int<33>)conv2_rx)))) && (((ap_int<33>)1 - ((ap_int<33>)conv2_ry)) <= ((ap_int<33>)(yy_reuse1 + -2)))) && (((ap_int<33>)(yy_reuse1 + -2)) < ((ap_int<33>)9 - ((ap_int<33>)conv2_ry)))) ? ((ap_uint<32>)(((65535U - ((ap_uint<32>)(WB1[0][0][conv2_ry][conv2_rx] ^ w_conv2[ff1][0][conv2_ry][conv2_rx])))[conv2_rb] << 1) - 1U)) : ((ap_uint<32>)0U))) + ((ap_int<34>)conv2_sum)));
                }
              }
            }
            conv2[0][ff1][(yy_reuse1 + -2)][(xx_reuse1 + -2)] = conv2_sum;
          }
        }
      }
    }
  }
  ap_uint<32> bn2[1][1][8][8];
  #pragma HLS array_partition variable=bn2 complete dim=4
  ap_int<32> bn2_partitioned;
LOOP7: for (ap_int<32> h2 = 0; h2 < 8; ++h2) {
    for (ap_int<32> w2 = 0; w2 < 8; ++w2) {
    #pragma HLS pipeline
      ap_uint<32> bn2_pack;
      bn2_pack = 0U;
      for (ap_int<32> i4 = 0; i4 < 32; ++i4) {
        bn2_pack(i4, i4) = ((bn_t2[i4][h2][w2] < ((ap_fixed<20, 10>)conv2[0][i4][h2][w2])) ? ((ap_int<32>)1) : ((ap_int<32>)0));
      }
      bn2[0][0][h2][w2] = bn2_pack;
    }
  }
  ap_uint<32> maxpool2[1][1][4][4];
  #pragma HLS array_partition variable=maxpool2 complete dim=4
  ap_int<32> maxpool2_partitioned;
LOOP8: for (ap_int<32> i5 = 0; i5 < 1; ++i5) {
    for (ap_int<32> h3 = 0; h3 < 4; ++h3) {
    #pragma HLS pipeline
      for (ap_int<32> w3 = 0; w3 < 4; ++w3) {
        maxpool2[i5][0][h3][w3] = (((bn2[i5][0][((w3 / 4) + (h3 * 2))][((((w3 + (h3 * 8)) + (i5 * 32)) * 2) % 8)] | bn2[0][0][(((((w3 + (h3 * 8)) + (i5 * 32)) * 2) + 1) / 8)][(((((w3 + (h3 * 8)) + (i5 * 32)) * 2) + 1) % 8)]) | bn2[i5][0][(((w3 / 4) + (h3 * 2)) + 1)][((((w3 + (h3 * 8)) + (i5 * 32)) * 2) % 8)]) | bn2[0][0][((((((w3 + (h3 * 8)) + (i5 * 32)) * 2) - (((((w3 + (h3 * 8)) + (i5 * 32)) * 2) + 1) % 8)) + 9) / 8)][(((((w3 + (h3 * 8)) + (i5 * 32)) * 2) + 1) % 8)]);
      }
    }
  }
  ap_int<32> packed_flatten[1][16];
LOOP9: for (ap_int<32> i6 = 0; i6 < 1; ++i6) {
    for (ap_int<32> j = 0; j < 16; ++j) {
    #pragma HLS pipeline
      packed_flatten[i6][j] = ((ap_int<32>)maxpool2[i6][0][(j / 4)][(j % 4)]);
    }
  }
  ap_int<32> fc1_matmul[1][256];
LOOP10: for (ap_int<32> j1 = 0; j1 < 256; ++j1) {
    ap_int<32> fc1_popcnt;
    fc1_popcnt = 0;
    for (ap_int<32> fc1_rk = 0; fc1_rk < 16; ++fc1_rk) {
    #pragma HLS pipeline
      for (ap_int<32> fc1_rb = 0; fc1_rb < 32; ++fc1_rb) {
        fc1_popcnt = ((ap_int<32>)(((ap_int<33>)(packed_flatten[0][fc1_rk] ^ w_fc1[j1][fc1_rk])[fc1_rb]) + ((ap_int<33>)fc1_popcnt)));
      }
    }
    fc1_matmul[0][j1] = fc1_popcnt;
  }
  ap_int<32> fc1[1][8];
LOOP11: for (ap_int<32> i7 = 0; i7 < 1; ++i7) {
    for (ap_int<32> j2 = 0; j2 < 8; ++j2) {
    #pragma HLS pipeline
      ap_int<32> fc1_pack;
      fc1_pack = 0;
      for (ap_int<32> i8 = 0; i8 < 32; ++i8) {
        fc1_pack(i8, i8) = ((0.000000e+00f < ((((float)(512 - (fc1_matmul[i7][((j2 * 32) + i8)] << 1))) * 3.535534e-01f) + ((float)b_fc1[((j2 * 32) + i8)]))) ? ((ap_int<32>)1) : ((ap_int<32>)0));
      }
      fc1[i7][j2] = fc1_pack;
    }
  }
  ap_int<32> fc2_matmul[1][10];
LOOP12: for (ap_int<32> j3 = 0; j3 < 10; ++j3) {
    ap_int<32> fc2_popcnt;
    fc2_popcnt = 0;
    for (ap_int<32> fc2_rk = 0; fc2_rk < 8; ++fc2_rk) {
    #pragma HLS pipeline
      for (ap_int<32> fc2_rb = 0; fc2_rb < 32; ++fc2_rb) {
        fc2_popcnt = ((ap_int<32>)(((ap_int<33>)(fc1[0][fc2_rk] ^ w_fc2[j3][fc2_rk])[fc2_rb]) + ((ap_int<33>)fc2_popcnt)));
      }
    }
    fc2_matmul[0][j3] = fc2_popcnt;
  }
LOOP13: for (ap_int<32> i9 = 0; i9 < 1; ++i9) {
    for (ap_int<32> j4 = 0; j4 < 10; ++j4) {
    #pragma HLS pipeline
      fc2[i9][j4] = ((ap_fixed<20, 10>)((((float)(256 - (fc2_matmul[i9][j4] << 1))) * 5.000000e-01f) + ((float)b_fc2[j4])));
    }
  }
}



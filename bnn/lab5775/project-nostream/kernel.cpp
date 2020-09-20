// HASH:1235494958
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
#include "const.h"


extern "C" {
void test(ap_uint<8> input_image[1][16][16][1], ap_fixed<32, 22> fc2[1][10]) {
    #pragma HLS INTERFACE m_axi port=input_image offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=fc2 offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=input_image bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2 bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
      #pragma HLS array_partition variable=input_image complete dim=3
      ap_uint<8> conv1_pad[1][18][18][1];
      #pragma HLS array_partition variable=conv1_pad complete dim=3
      ap_int<32> conv1_pad_partitioned;
      conv1_pad_hh: for (ap_int<32> hh = 1; hh < 17; ++hh) {
      #pragma HLS pipeline
        conv1_pad_ww: for (ap_int<32> ww = 1; ww < 17; ++ww) {
          conv1_pad[0][hh][ww][0] = (ap_uint<8>)input_image[0][(hh + -1)][(ww + -1)][0];//((ap_uint<8>)(((((1 <= ww) && (ww < 17)) && (1 <= hh)) && (hh < 17)) ? (((ap_uint<32>)input_image[0][(hh + -1)][(ww + -1)][0])) : ((ap_uint<32>)0U)));
        }
      }
      ap_int<6> conv1[1][16][16][16];
      #pragma HLS array_partition variable=conv1 complete dim=4
      ap_uint<8> LB1[1][3][18][1];
      ap_uint<8> WB1[1][3][3][1];
      ap_int<32> conv1_partitioned;
      conv1_yy_reuse: for (ap_int<32> yy_reuse = 0; yy_reuse < 18; ++yy_reuse) {
        conv1_xx_reuse: for (ap_int<32> xx_reuse = 0; xx_reuse < 18; ++xx_reuse) {
        #pragma HLS pipeline
          conv1_pad_2: for (ap_int<32> conv1_pad_2 = 0; conv1_pad_2 < 2; ++conv1_pad_2) {
            LB1[0][conv1_pad_2][xx_reuse][0] = LB1[0][(conv1_pad_2 + 1)][xx_reuse][0];
          }
          LB1[0][2][xx_reuse][0] = conv1_pad[0][yy_reuse][xx_reuse][0];
          if (2 <= yy_reuse) {
            LB1_2: for (ap_int<32> LB1_2 = 0; LB1_2 < 3; ++LB1_2) {
              LB1_1: for (ap_int<32> LB1_1 = 0; LB1_1 < 2; ++LB1_1) {
                WB1[0][LB1_2][LB1_1][0] = WB1[0][LB1_2][(LB1_1 + 1)][0];
              }
              WB1[0][LB1_2][2][0] = LB1[0][LB1_2][xx_reuse][0];
            }
              if (2 <= xx_reuse) {
            conv1_ff: for (ap_int<32> ff = 0; ff < 16; ++ff) {
                ap_int<6> conv1_sum;
                conv1_sum = (ap_int<6>)0;
                conv1_conv1_ry: for (ap_int<32> conv1_ry = 0; conv1_ry < 3; ++conv1_ry) {
                  conv1_conv1_rx: for (ap_int<32> conv1_rx = 0; conv1_rx < 3; ++conv1_rx) {
                    conv1_conv1_rb: for (ap_int<32> conv1_rb = 0; conv1_rb < 8; ++conv1_rb) {
                      conv1_sum = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)conv1_rx)) <= ((ap_int<33>)(xx_reuse + -2))) && (((ap_int<33>)(xx_reuse + -2)) < ((ap_int<33>)17 - ((ap_int<33>)conv1_rx)))) && (((ap_int<33>)1 - ((ap_int<33>)conv1_ry)) <= ((ap_int<33>)(yy_reuse + -2)))) && (((ap_int<33>)(yy_reuse + -2)) < ((ap_int<33>)17 - ((ap_int<33>)conv1_ry)))) ? ((ap_uint<32>)(((255U - ((ap_uint<32>)(WB1[0][conv1_ry][conv1_rx][0] ^ w_conv1[ff][conv1_ry][conv1_rx][0])))[conv1_rb] << 1) - 1U)) : ((ap_uint<32>)0U))) + ((ap_int<34>)conv1_sum)));
                    }
                  }
                }
                conv1[0][(yy_reuse + -2)][(xx_reuse + -2)][ff] = conv1_sum;
              }
            }
          }
        }
      }
      ap_uint<16> bn1[1][16][16][1];
      #pragma HLS array_partition variable=bn1 complete dim=3
      ap_int<32> bn1_partitioned;
      bn1_h: for (ap_int<32> h = 0; h < 16; ++h) {
        bn1_w: for (ap_int<32> w = 0; w < 16; ++w) {
        #pragma HLS pipeline
          ap_uint<16> bn1_pack;
          bn1_pack = (ap_uint<16>)0;
          i: for (ap_int<32> i = 0; i < 16; ++i) {
            bn1_pack(i, i) = ((bn_t1[h][w][i] < ((ap_fixed<20, 10>)conv1[0][h][w][i])) ? ((ap_int<32>)1) : ((ap_int<32>)0));
          }
          bn1[0][h][w][0] = bn1_pack;
        }
      }
      ap_uint<16> maxpool1[1][8][8][1];
      #pragma HLS array_partition variable=maxpool1 complete dim=3
      ap_int<32> maxpool1_partitioned;
      maxpool1_h1: for (ap_int<32> h1 = 0; h1 < 8; ++h1) {
      #pragma HLS pipeline
        maxpool1_w1: for (ap_int<32> w1 = 0; w1 < 8; ++w1) {
          ap_uint<16> reducer0;
          reducer0 = (ap_uint<16>)0;
          maxpool1_ra0: for (ap_int<32> ra0 = 0; ra0 < 2; ++ra0) {
            maxpool1_ra1: for (ap_int<32> ra1 = 0; ra1 < 2; ++ra1) {
              reducer0 = (bn1[0][((h1 * 2) + ra0)][((w1 * 2) + ra1)][0] | reducer0);
            }
          }
          maxpool1[0][h1][w1][0] = reducer0;
        }
      }
      ap_uint<16> conv2_pad[1][10][10][1];
      #pragma HLS array_partition variable=conv2_pad complete dim=3
      ap_int<32> conv2_pad_partitioned;
      conv2_pad_hh1: for (ap_int<32> hh1 = 1; hh1 < 9; ++hh1) {
      #pragma HLS pipeline
        conv2_pad_ww1: for (ap_int<32> ww1 = 1; ww1 < 9; ++ww1) {
          conv2_pad[0][hh1][ww1][0] = (ap_uint<16>)maxpool1[0][(hh1 + -1)][(ww1 + -1)][0];//((ap_uint<16>)(((((1 <= ww1) && (ww1 < 9)) && (1 <= hh1)) && (hh1 < 9)) ? (((ap_uint<32>)maxpool1[0][(hh1 + -1)][(ww1 + -1)][0])) : ((ap_uint<32>)0U)));
        }
      }
      ap_int<6> conv2[1][8][8][32];
      #pragma HLS array_partition variable=conv2 complete dim=4
      ap_uint<16> LB2[1][3][10][1];
      ap_uint<16> WB2[1][3][3][1];
      ap_int<32> conv2_partitioned;
      conv2_yy_reuse1: for (ap_int<32> yy_reuse1 = 0; yy_reuse1 < 10; ++yy_reuse1) {
        conv2_xx_reuse1: for (ap_int<32> xx_reuse1 = 0; xx_reuse1 < 10; ++xx_reuse1) {
        #pragma HLS pipeline
          conv2_pad_2: for (ap_int<32> conv2_pad_2 = 0; conv2_pad_2 < 2; ++conv2_pad_2) {
            LB2[0][conv2_pad_2][xx_reuse1][0] = LB2[0][(conv2_pad_2 + 1)][xx_reuse1][0];
          }
          LB2[0][2][xx_reuse1][0] = conv2_pad[0][yy_reuse1][xx_reuse1][0];
          if (2 <= yy_reuse1) {
            LB2_2: for (ap_int<32> LB2_2 = 0; LB2_2 < 3; ++LB2_2) {
              LB2_1: for (ap_int<32> LB2_1 = 0; LB2_1 < 2; ++LB2_1) {
                WB2[0][LB2_2][LB2_1][0] = WB2[0][LB2_2][(LB2_1 + 1)][0];
              }
              WB2[0][LB2_2][2][0] = LB2[0][LB2_2][xx_reuse1][0];
            }
              if (2 <= xx_reuse1) {
            conv2_ff1: for (ap_int<32> ff1 = 0; ff1 < 32; ++ff1) {
                ap_int<6> conv2_sum;
                conv2_sum = (ap_int<6>)0;
                conv2_conv2_ry: for (ap_int<32> conv2_ry = 0; conv2_ry < 3; ++conv2_ry) {
                  conv2_conv2_rx: for (ap_int<32> conv2_rx = 0; conv2_rx < 3; ++conv2_rx) {
                    conv2_conv2_rb: for (ap_int<32> conv2_rb = 0; conv2_rb < 16; ++conv2_rb) {
                      conv2_sum = ((ap_int<6>)(((ap_int<34>)(((((((ap_int<33>)1 - ((ap_int<33>)conv2_rx)) <= ((ap_int<33>)(xx_reuse1 + -2))) && (((ap_int<33>)(xx_reuse1 + -2)) < ((ap_int<33>)9 - ((ap_int<33>)conv2_rx)))) && (((ap_int<33>)1 - ((ap_int<33>)conv2_ry)) <= ((ap_int<33>)(yy_reuse1 + -2)))) && (((ap_int<33>)(yy_reuse1 + -2)) < ((ap_int<33>)9 - ((ap_int<33>)conv2_ry)))) ? ((ap_uint<32>)(((65535U - ((ap_uint<32>)(WB2[0][conv2_ry][conv2_rx][0] ^ w_conv2[ff1][conv2_ry][conv2_rx][0])))[conv2_rb] << 1) - 1U)) : ((ap_uint<32>)0U))) + ((ap_int<34>)conv2_sum)));
                    }
                  }
                }
                conv2[0][(yy_reuse1 + -2)][(xx_reuse1 + -2)][ff1] = conv2_sum;
              }
            }
          }
        }
      }
      ap_uint<32> bn2[1][8][8][1];
      #pragma HLS array_partition variable=bn2 complete dim=3
      ap_int<32> bn2_partitioned;
      bn2_h2: for (ap_int<32> h2 = 0; h2 < 8; ++h2) {
        bn2_w2: for (ap_int<32> w2 = 0; w2 < 8; ++w2) {
        #pragma HLS pipeline
          ap_uint<32> bn2_pack;
          bn2_pack = 0U;
          i1: for (ap_int<32> i1 = 0; i1 < 32; ++i1) {
            bn2_pack(i1, i1) = ((bn_t2[h2][w2][i1] < ((ap_fixed<20, 10>)conv2[0][h2][w2][i1])) ? ((ap_int<32>)1) : ((ap_int<32>)0));
          }
          bn2[0][h2][w2][0] = bn2_pack;
        }
      }
      ap_uint<32> maxpool2[1][4][4][1];
      #pragma HLS array_partition variable=maxpool2 complete dim=3
      ap_int<32> maxpool2_partitioned;
      maxpool2_h3: for (ap_int<32> h3 = 0; h3 < 4; ++h3) {
      #pragma HLS pipeline
        maxpool2_w3: for (ap_int<32> w3 = 0; w3 < 4; ++w3) {
          ap_uint<32> reducer1;
          reducer1 = 0U;
          maxpool2_ra2: for (ap_int<32> ra2 = 0; ra2 < 2; ++ra2) {
            maxpool2_ra3: for (ap_int<32> ra3 = 0; ra3 < 2; ++ra3) {
              reducer1 = (bn2[0][((h3 * 2) + ra2)][((w3 * 2) + ra3)][0] | reducer1);
            }
          }
          maxpool2[0][h3][w3][0] = reducer1;
        }
      }
      ap_int<32> packed_flatten[1][16];
      packed_flatten_j: for (ap_int<32> j = 0; j < 16; ++j) {
      #pragma HLS pipeline
        packed_flatten[0][j] = ((ap_int<32>)maxpool2[0][0][j][0]);
      }
      ap_int<32> fc1_matmul[1][256];
      fc1_matmul_j1: for (ap_int<32> j1 = 0; j1 < 256; ++j1) {
        ap_int<32> fc1_popcnt;
        fc1_popcnt = 0;
        fc1_matmul_fc1_rk: for (ap_int<32> fc1_rk = 0; fc1_rk < 16; ++fc1_rk) {
        #pragma HLS pipeline
          fc1_matmul_fc1_rb: for (ap_int<32> fc1_rb = 0; fc1_rb < 32; ++fc1_rb) {
            fc1_popcnt = ((ap_int<32>)(((ap_int<33>)(packed_flatten[0][fc1_rk] ^ w_fc1[j1][fc1_rk])[fc1_rb]) + ((ap_int<33>)fc1_popcnt)));
          }
        }
        fc1_matmul[0][j1] = fc1_popcnt;
      }
      ap_int<32> fc1[1][8];
      fc1_j2: for (ap_int<32> j2 = 0; j2 < 8; ++j2) {
        ap_int<32> fc1_pack;
        fc1_pack = 0;
        i2: for (ap_int<32> i2 = 0; i2 < 32; ++i2) {
        #pragma HLS pipeline
          fc1_pack(i2, i2) = ((0.000000e+00f < ((((float)(512 - (fc1_matmul[0][((j2 * 32) + i2)] << 1))) * 3.535534e-01f) + ((float)b_fc1[((j2 * 32) + i2)]))) ? ((ap_int<32>)1) : ((ap_int<32>)0));
        }
        fc1[0][j2] = fc1_pack;
      }
      ap_int<32> fc2_matmul[1][10];
      fc2_matmul_j3: for (ap_int<32> j3 = 0; j3 < 10; ++j3) {
        ap_int<32> fc2_popcnt;
        fc2_popcnt = 0;
        fc2_matmul_fc2_rk: for (ap_int<32> fc2_rk = 0; fc2_rk < 8; ++fc2_rk) {
        #pragma HLS pipeline
          fc2_matmul_fc2_rb: for (ap_int<32> fc2_rb = 0; fc2_rb < 32; ++fc2_rb) {
            fc2_popcnt = ((ap_int<32>)(((ap_int<33>)(fc1[0][fc2_rk] ^ w_fc2[j3][fc2_rk])[fc2_rb]) + ((ap_int<33>)fc2_popcnt)));
          }
        }
        fc2_matmul[0][j3] = fc2_popcnt;
      }
      fc2_j4: for (ap_int<32> j4 = 0; j4 < 10; ++j4) {
      #pragma HLS pipeline
        fc2[0][j4] = ((ap_fixed<32, 22>)((((float)(256 - (fc2_matmul[0][j4] << 1))) * 5.000000e-01f) + ((float)b_fc2[j4])));
      }
    }
}


// HASH:2338388791
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
      ap_int<32> _top;
      ap_uint<8> conv1_pad[1][18][18][1];
      ap_uint<8> conv1_pad_pipe_1[1][18][18][1];
      #pragma HLS dataflow
      #pragma HLS stream variable=conv1_pad_pipe_1 depth=324
      conv1_pad_hh: for (ap_int<32> hh = 0; hh < 18; ++hh) {
        conv1_pad_ww: for (ap_int<32> ww = 0; ww < 18; ++ww) {
        #pragma HLS pipeline
          ap_uint<8> conv1_pad_temp;
          conv1_pad_temp = ((ap_uint<8>)(((((1 <= ww) && (ww < 17)) && (1 <= hh)) && (hh < 17)) ? (((ap_uint<32>)input_image[((((ww - ((ww + -1) % 16)) + (hh * 16)) + -17) / 256)][(((((ww - ((ww + -1) % 16)) + (hh * 16)) + -17) / 16) % 16)][((ww + -1) % 16)][0])) : ((ap_uint<32>)0U)));
          conv1_pad_pipe_1[0][hh][ww][0] = conv1_pad_temp;
          conv1_pad[0][hh][ww][0] = conv1_pad_temp;
        }
      }
      ap_int<6> conv1[1][16][16][16];
      ap_uint<8> LB1[1][3][18][1];
      ap_uint<8> WB1[1][3][3][1];
      ap_int<6> conv1_pipe_2[1][16][16][16];
      #pragma HLS stream variable=conv1_pipe_2 depth=4096
      conv1_yy_reuse: for (ap_int<32> yy_reuse = 0; yy_reuse < 18; ++yy_reuse) {
        conv1_xx_reuse: for (ap_int<32> xx_reuse = 0; xx_reuse < 18; ++xx_reuse) {
        #pragma HLS pipeline
          conv1_pad_2: for (ap_int<32> conv1_pad_2 = 0; conv1_pad_2 < 2; ++conv1_pad_2) {
            LB1[0][conv1_pad_2][xx_reuse][0] = LB1[0][(conv1_pad_2 + 1)][xx_reuse][0];
          }
          ap_uint<8> conv1_pad_temp1;
          conv1_pad_temp1 = conv1_pad_pipe_1[0][yy_reuse][xx_reuse][0];
          LB1[0][2][xx_reuse][0] = conv1_pad_temp1;
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
                ap_int<6> conv1_temp;
                conv1_temp = conv1_sum;
                conv1_pipe_2[0][(yy_reuse + -2)][(xx_reuse + -2)][ff] = conv1_temp;
              }
            }
          }
        }
      }
      ap_uint<16> bn1[1][16][16][1];
      ap_uint<16> bn1_pipe_3[1][16][16][1];
      #pragma HLS stream variable=bn1_pipe_3 depth=256
      bn1_h: for (ap_int<32> h = 0; h < 16; ++h) {
        bn1_w: for (ap_int<32> w = 0; w < 16; ++w) {
        #pragma HLS pipeline
          ap_uint<16> bn1_pack;
          bn1_pack = (ap_uint<16>)0;
          i: for (ap_int<32> i = 0; i < 16; ++i) {
            ap_int<6> conv1_temp1;
            conv1_temp1 = conv1_pipe_2[0][h][w][i];
            bn1_pack(i, i) = ((bn_t1[h][w][i] < ((ap_fixed<20, 10>)conv1_temp1)) ? ((ap_int<32>)1) : ((ap_int<32>)0));
          }
          ap_uint<16> bn1_temp;
          bn1_temp = bn1_pack;
          bn1_pipe_3[0][h][w][0] = bn1_temp;
        }
      }
      ap_uint<16> maxpool1_res[1][8][8][1];
      ap_uint<16> maxpool1_LB[2][16];
      ap_int<32> maxpool1;
      ap_uint<16> maxpool1_res_pipe_4[1][8][8][1];
      #pragma HLS stream variable=maxpool1_res_pipe_4 depth=64
      maxpool1_hh1: for (ap_int<32> hh1 = 0; hh1 < 8; ++hh1) {
        maxpool1_LB_i: for (ap_int<32> maxpool1_LB_i = 0; maxpool1_LB_i < 2; ++maxpool1_LB_i) {
        #pragma HLS pipeline
          maxpool1_LB_j: for (ap_int<32> maxpool1_LB_j = 0; maxpool1_LB_j < 16; ++maxpool1_LB_j) {
            ap_uint<16> bn1_temp1;
            bn1_temp1 = bn1_pipe_3[0][((hh1 * 2) + maxpool1_LB_i)][maxpool1_LB_j][0];
            maxpool1_LB[maxpool1_LB_i][maxpool1_LB_j] = bn1_temp1;
          }
        }
        i1: for (ap_int<32> i1 = 0; i1 < 8; ++i1) {
          ap_uint<16> maxpool1_val;
          maxpool1_val = (ap_uint<16>)0;
          maxpool1_ry: for (ap_int<32> maxpool1_ry = 0; maxpool1_ry < 2; ++maxpool1_ry) {
            maxpool1_rx: for (ap_int<32> maxpool1_rx = 0; maxpool1_rx < 2; ++maxpool1_rx) {
              maxpool1_val = (maxpool1_val | maxpool1_LB[maxpool1_ry][((i1 * 2) + maxpool1_rx)]);
            }
          }
          ap_uint<16> maxpool1_res_temp;
          maxpool1_res_temp = maxpool1_val;
          maxpool1_res_pipe_4[0][hh1][i1][0] = maxpool1_res_temp;
          maxpool1_res[0][hh1][i1][0] = maxpool1_res_temp;
        }
      }
      ap_uint<16> conv2_pad[1][10][10][1];
      ap_uint<16> conv2_pad_pipe_5[1][10][10][1];
      #pragma HLS stream variable=conv2_pad_pipe_5 depth=100
      conv2_pad_hh2: for (ap_int<32> hh2 = 0; hh2 < 10; ++hh2) {
        conv2_pad_ww1: for (ap_int<32> ww1 = 0; ww1 < 10; ++ww1) {
        #pragma HLS pipeline
          ap_uint<16> maxpool1_res_temp1;
          maxpool1_res_temp1 = maxpool1_res_pipe_4[((((ww1 - ((ww1 + -1) % 8)) + (hh2 * 8)) + -9) / 64)][(((((ww1 - ((ww1 + -1) % 8)) + (hh2 * 8)) + -9) / 8) % 8)][((ww1 + -1) % 8)][0];
          ap_uint<16> conv2_pad_temp;
          conv2_pad_temp = ((ap_uint<16>)(((((1 <= ww1) && (ww1 < 9)) && (1 <= hh2)) && (hh2 < 9)) ? (((ap_uint<32>)maxpool1_res_temp1)) : ((ap_uint<32>)0U)));
          conv2_pad_pipe_5[0][hh2][ww1][0] = conv2_pad_temp;
          conv2_pad[0][hh2][ww1][0] = conv2_pad_temp;
        }
      }
      ap_int<6> conv2[1][8][8][32];
      ap_uint<16> LB2[1][3][10][1];
      ap_uint<16> WB2[1][3][3][1];
      ap_int<6> conv2_pipe_6[1][8][8][32];
      #pragma HLS stream variable=conv2_pipe_6 depth=2048
      conv2_yy_reuse1: for (ap_int<32> yy_reuse1 = 0; yy_reuse1 < 10; ++yy_reuse1) {
        conv2_xx_reuse1: for (ap_int<32> xx_reuse1 = 0; xx_reuse1 < 10; ++xx_reuse1) {
        #pragma HLS pipeline
          conv2_pad_2: for (ap_int<32> conv2_pad_2 = 0; conv2_pad_2 < 2; ++conv2_pad_2) {
            LB2[0][conv2_pad_2][xx_reuse1][0] = LB2[0][(conv2_pad_2 + 1)][xx_reuse1][0];
          }
          ap_uint<16> conv2_pad_temp1;
          conv2_pad_temp1 = conv2_pad_pipe_5[0][yy_reuse1][xx_reuse1][0];
          LB2[0][2][xx_reuse1][0] = conv2_pad_temp1;
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
                ap_int<6> conv2_temp;
                conv2_temp = conv2_sum;
                conv2_pipe_6[0][(yy_reuse1 + -2)][(xx_reuse1 + -2)][ff1] = conv2_temp;
              }
            }
          }
        }
      }
      ap_uint<32> bn2[1][8][8][1];
      ap_uint<32> bn2_pipe_7[1][8][8][1];
      #pragma HLS stream variable=bn2_pipe_7 depth=64
      bn2_h1: for (ap_int<32> h1 = 0; h1 < 8; ++h1) {
        bn2_w1: for (ap_int<32> w1 = 0; w1 < 8; ++w1) {
        #pragma HLS pipeline
          ap_uint<32> bn2_pack;
          bn2_pack = 0U;
          i2: for (ap_int<32> i2 = 0; i2 < 32; ++i2) {
            ap_int<6> conv2_temp1;
            conv2_temp1 = conv2_pipe_6[0][h1][w1][i2];
            bn2_pack(i2, i2) = ((bn_t2[h1][w1][i2] < ((ap_fixed<20, 10>)conv2_temp1)) ? ((ap_int<32>)1) : ((ap_int<32>)0));
          }
          ap_uint<32> bn2_temp;
          bn2_temp = bn2_pack;
          bn2_pipe_7[0][h1][w1][0] = bn2_temp;
        }
      }
      ap_uint<32> maxpool2_res[1][4][4][1];
      ap_uint<32> maxpool2_LB[2][8];
      ap_int<32> maxpool2;
      ap_uint<32> maxpool2_res_pipe_8[1][4][4][1];
      #pragma HLS stream variable=maxpool2_res_pipe_8 depth=16
      maxpool2_hh3: for (ap_int<32> hh3 = 0; hh3 < 4; ++hh3) {
        maxpool2_LB_i: for (ap_int<32> maxpool2_LB_i = 0; maxpool2_LB_i < 2; ++maxpool2_LB_i) {
        #pragma HLS pipeline
          maxpool2_LB_j: for (ap_int<32> maxpool2_LB_j = 0; maxpool2_LB_j < 8; ++maxpool2_LB_j) {
            ap_uint<32> bn2_temp1;
            bn2_temp1 = bn2_pipe_7[0][((hh3 * 2) + maxpool2_LB_i)][maxpool2_LB_j][0];
            maxpool2_LB[maxpool2_LB_i][maxpool2_LB_j] = bn2_temp1;
          }
        }
        i3: for (ap_int<32> i3 = 0; i3 < 4; ++i3) {
          ap_uint<32> maxpool2_val;
          maxpool2_val = 0U;
          maxpool2_ry: for (ap_int<32> maxpool2_ry = 0; maxpool2_ry < 2; ++maxpool2_ry) {
            maxpool2_rx: for (ap_int<32> maxpool2_rx = 0; maxpool2_rx < 2; ++maxpool2_rx) {
              maxpool2_val = (maxpool2_val | maxpool2_LB[maxpool2_ry][((i3 * 2) + maxpool2_rx)]);
            }
          }
          ap_uint<32> maxpool2_res_temp;
          maxpool2_res_temp = maxpool2_val;
          maxpool2_res_pipe_8[0][hh3][i3][0] = maxpool2_res_temp;
          maxpool2_res[0][hh3][i3][0] = maxpool2_res_temp;
        }
      }
      ap_int<32> packed_flatten[1][16];
      ap_int<32> packed_flatten_pipe_9[1][16];
      #pragma HLS stream variable=packed_flatten_pipe_9 depth=16
      packed_flatten_j: for (ap_int<32> j = 0; j < 16; ++j) {
      #pragma HLS pipeline
        ap_uint<32> maxpool2_res_temp1;
        maxpool2_res_temp1 = maxpool2_res_pipe_8[0][(j / 4)][(j % 4)][0];
        ap_int<32> packed_flatten_temp;
        packed_flatten_temp = ((ap_int<32>)maxpool2_res_temp1);
        packed_flatten_pipe_9[0][j] = packed_flatten_temp;
      }
      ap_int<32> fc1_matmul[1][256];
      ap_int<32> fc1_LB[16];
      #pragma HLS array_partition variable=fc1_LB complete dim=0
      ap_int<32> fc1_matmul_pipe_10[1][256];
      #pragma HLS stream variable=fc1_matmul_pipe_10 depth=256
      fc1_matmul_j1: for (ap_int<32> j1 = 0; j1 < 256; ++j1) {
      #pragma HLS pipeline
        ap_int<32> fc1_popcnt;
        fc1_popcnt = 0;
        if (j1 == 0) {
          fc1_init: for (ap_int<32> i = 0; i < 16; ++i) {
            fc1_LB[i] = packed_flatten_pipe_9[0][i];
          }
        }
        fc1_matmul_fc1_rk: for (ap_int<32> fc1_rk = 0; fc1_rk < 16; ++fc1_rk) {
          fc1_matmul_fc1_rb: for (ap_int<32> fc1_rb = 0; fc1_rb < 32; ++fc1_rb) {
            ap_int<32> packed_flatten_temp1;
            packed_flatten_temp1 = fc1_LB[fc1_rk];
            fc1_popcnt = ((ap_int<32>)(((ap_int<33>)(packed_flatten_temp1 ^ w_fc1[j1][fc1_rk])[fc1_rb]) + ((ap_int<33>)fc1_popcnt)));
          }
        }
        ap_int<32> fc1_matmul_temp;
        fc1_matmul_temp = fc1_popcnt;
        fc1_matmul_pipe_10[0][j1] = fc1_matmul_temp;
      }
      ap_int<32> fc1[1][8];
      ap_int<32> fc1_pipe_11[1][8];
      #pragma HLS stream variable=fc1_pipe_11 depth=8
      fc1_j2: for (ap_int<32> j2 = 0; j2 < 8; ++j2) {
        ap_int<32> fc1_pack;
        fc1_pack = 0;
        i4: for (ap_int<32> i4 = 0; i4 < 32; ++i4) {
        #pragma HLS pipeline
          ap_int<32> fc1_matmul_temp1;
          fc1_matmul_temp1 = fc1_matmul_pipe_10[0][((j2 * 32) + i4)];
          fc1_pack(i4, i4) = ((0.000000e+00f < ((((float)(512 - (fc1_matmul_temp1 << 1))) * 3.535534e-01f) + ((float)b_fc1[((j2 * 32) + i4)]))) ? ((ap_int<32>)1) : ((ap_int<32>)0));
        }
        ap_int<32> fc1_temp;
        fc1_temp = fc1_pack;
        fc1_pipe_11[0][j2] = fc1_temp;
      }
      ap_int<32> fc2_matmul[1][10];
      ap_int<32> fc2_LB[8];
      #pragma HLS array_partition variable=fc2_LB complete dim=0
      ap_int<32> fc2_matmul_pipe_12[1][10];
      #pragma HLS stream variable=fc2_matmul_pipe_12 depth=10
      fc2_matmul_j3: for (ap_int<32> j3 = 0; j3 < 10; ++j3) {
      #pragma HLS pipeline
        ap_int<32> fc2_popcnt;
        fc2_popcnt = 0;
        if (j3 == 0) {
          f2_init: for (ap_int<32> i = 0; i < 8; ++i) {
            fc2_LB[i] = fc1_pipe_11[0][i];
          }
        }
        fc2_matmul_fc2_rk: for (ap_int<32> fc2_rk = 0; fc2_rk < 8; ++fc2_rk) {
          fc2_matmul_fc2_rb: for (ap_int<32> fc2_rb = 0; fc2_rb < 32; ++fc2_rb) {
            ap_int<32> fc1_temp1;
            fc1_temp1 = fc2_LB[fc2_rk];
            fc2_popcnt = ((ap_int<32>)(((ap_int<33>)(fc1_temp1 ^ w_fc2[j3][fc2_rk])[fc2_rb]) + ((ap_int<33>)fc2_popcnt)));
          }
        }
        ap_int<32> fc2_matmul_temp;
        fc2_matmul_temp = fc2_popcnt;
        fc2_matmul_pipe_12[0][j3] = fc2_matmul_temp;
      }
      fc2_j4: for (ap_int<32> j4 = 0; j4 < 10; ++j4) {
      #pragma HLS pipeline
        ap_int<32> fc2_matmul_temp1;
        fc2_matmul_temp1 = fc2_matmul_pipe_12[0][j4];
        fc2[0][j4] = ((ap_fixed<32, 22>)((((float)(256 - (fc2_matmul_temp1 << 1))) * 5.000000e-01f) + ((float)b_fc2[j4])));
      }
    }
}


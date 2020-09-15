//
#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>

//const static uint4 lut16[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
const static uint32 m1("0x55555555", 16);
const static uint32 m2("0x33333333", 16);
const static uint32 m4("0x0f0f0f0f", 16);

inline uint6 compute_engine_64(uint64 b, uint64 w)
{
#pragma HLS latency max=0
    uint64 x = ~(b^w);

    x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
    x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits
    x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits
    x += x >>  8;  //put count of each 16 bits into their lowest 8 bits
    x += x >> 16;  //put count of each 32 bits into their lowest 8 bits
    x += x >> 32;  //put count of each 64 bits into their lowest 8 bits
    return x & 0x7f;
}

// inline uint6 compute_engine_64(uint64 b, uint64 w)
// {
// #pragma HLS PIPELINE
//     uint64 t = ~(b^w);
//     ap_int<4> add0, add1, add2, add3, add4, add5, add6, add7;
//     ap_int<5> add8, add9, adda, addb;
//     ap_int<6> addc, addd;
//
//     add0 = lut16[(int)t.range(3,  0 )] + lut16[(int)t.range(7,  4 )];
//     add1 = lut16[(int)t.range(11, 8 )] + lut16[(int)t.range(15, 12)];
//     add2 = lut16[(int)t.range(19, 16)] + lut16[(int)t.range(23, 20)];
//     add3 = lut16[(int)t.range(27, 24)] + lut16[(int)t.range(31, 28)];
//     add4 = lut16[(int)t.range(35, 32)] + lut16[(int)t.range(39, 36)];
//     add5 = lut16[(int)t.range(43, 40)] + lut16[(int)t.range(47, 44)];
//     add6 = lut16[(int)t.range(51, 48)] + lut16[(int)t.range(55, 52)];
//     add7 = lut16[(int)t.range(59, 56)] + lut16[(int)t.range(63, 60)];
//
//     add8 = add0 + add1;
//     add9 = add2 + add3;
//     adda = add4 + add5;
//     addb = add6 + add7;
//
//     addc = add8 + add9;
//     addd = adda + addb;
//
//     return addc + addd;
// }

inline FIX_FM_acc batch_norm(uint8 sum, FIX_WT weight, FIX_WT bias)
{
    return sum*weight + bias;
}

inline FIX_FM_acc relu(FIX_FM_acc norm, FIX_WT shiftx, FIX_WT shifty, FIX_WT weight)
{
    if (norm > 0) {
        return norm + shifty;
    } else {
        return norm*weight + shifty;
    }
}


inline uint8 sum_engine(uint6 t0,
        uint6 t1,
        uint6 t2,
        uint6 t3,
        uint6 t4,
        uint6 t5,
        uint6 t6,
        uint6 t7,
        uint6 t8)
{
#pragma HLS PIPELINE
    ap_int<6> add0, add1, add2, add3;
    ap_int<7> add4, add5, add6;

    add0 = t0 + t1;
    add1 = t2 + t3;
    add2 = t4 + t5;
    add3 = t6 + t7;

    add4 = add0 + add1;
    add5 = add2 + add3;

    add6 = add4 + add5;

    return add6 + t8;
}

void pgconv64_1bit(uint64 bottom1[9][9],
//                uint64 bottom0[9][9],
                uint64 weights[32][3][3],
                FIX_WT thres[32],
//                FIX_WT bn_weights[32],
//                FIX_WT bn_bias[32],
//                FIX_WT relu_shiftx[32],
//                FIX_WT relu_shifty[32],
//                FIX_WT relu_weights[32],
                FIX_FM_acc top[32][9][9],
				int stride
)
{

#pragma HLS array_partition variable=weights dim=0 complete
#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
#pragma HLS array_partition variable=top dim=1 complete

    uint64 bot1_LB[3][9];
    uint64 bot1_WB[3][3];
    uint64 bot0_LB[3][9];
    uint64 bot0_WB[3][3];
	FIX_FM_acc d[16];
#pragma HLS array_partition variable=d dim=1 complete
	for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
		d[i] = top[0*16 + i][0][0];
	}
    biconv_row:for(int row = 1; row < 8; row ++){
        biconv_col:for(int col = 1; col < 8; col ++) {
        	for (int c = 0; c < 2; c++) {
#pragma HLS PIPELINE
				bot1_LB[0][col] = bot1_LB[1][col];
				bot1_LB[1][col] = bot1_LB[2][col];
				bot1_LB[2][col] = bottom1[row][col];
//				bot0_LB[0][col] = bot0_LB[1][col];
//				bot0_LB[1][col] = bot0_LB[2][col];
//				bot0_LB[2][col] = bottom0[row][col];
				if (0 < row && row < 8) { // move left one column, update window buffer
					for (int LB_1 = 0; LB_1 < 3; ++LB_1) {
#pragma HLS UNROLL
						bot1_WB[LB_1][0] = bot1_WB[LB_1][1];
						bot1_WB[LB_1][1] = bot1_WB[LB_1][2];
						bot1_WB[LB_1][2] = bot1_LB[LB_1][col];
//						bot0_WB[LB_1][0] = bot0_WB[LB_1][1];
//						bot0_WB[LB_1][1] = bot0_WB[LB_1][2];
//						bot0_WB[LB_1][2] = bot0_LB[LB_1][col];
					}
					if (0 < col && col < 8) {
						biconv_coo:for (int coo = 0; coo < 16; coo ++) {
#pragma HLS UNROLL
							int w_i = c*16 + coo;
							uint6 tmp0 = compute_engine_64(bot1_WB[0][0], weights[w_i][0][0]);
							uint6 tmp1 = compute_engine_64(bot1_WB[0][1], weights[w_i][0][1]);
							uint6 tmp2 = compute_engine_64(bot1_WB[0][2], weights[w_i][0][2]);
							uint6 tmp3 = compute_engine_64(bot1_WB[1][0], weights[w_i][1][0]);
							uint6 tmp4 = compute_engine_64(bot1_WB[1][1], weights[w_i][1][1]);
							uint6 tmp5 = compute_engine_64(bot1_WB[1][2], weights[w_i][1][2]);
							uint6 tmp6 = compute_engine_64(bot1_WB[2][0], weights[w_i][2][0]);
							uint6 tmp7 = compute_engine_64(bot1_WB[2][1], weights[w_i][2][1]);
							uint6 tmp8 = compute_engine_64(bot1_WB[2][2], weights[w_i][2][2]);
							uint8 sum = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);

//                if (sum > thres[coo]) {
//                  uint6 tmp0 = compute_engine_16(bottom0[row-1][col-1], weights[coo][0][0]);
//                  uint6 tmp1 = compute_engine_16(bottom0[row-1][col  ], weights[coo][0][1]);
//                  uint6 tmp2 = compute_engine_16(bottom0[row-1][col+1], weights[coo][0][2]);
//                  uint6 tmp3 = compute_engine_16(bottom0[row  ][col-1], weights[coo][1][0]);
//                  uint6 tmp4 = compute_engine_16(bottom0[row  ][col  ], weights[coo][1][1]);
//                  uint6 tmp5 = compute_engine_16(bottom0[row  ][col+1], weights[coo][1][2]);
//                  uint6 tmp6 = compute_engine_16(bottom0[row+1][col-1], weights[coo][2][0]);
//                  uint6 tmp7 = compute_engine_16(bottom0[row+1][col  ], weights[coo][2][1]);
//                  uint6 tmp8 = compute_engine_16(bottom0[row+1][col+1], weights[coo][2][2]);
//                  sum += sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
//                }

//                uint6 tmp00 = compute_engine_64(bottom0[row-1][col-1], weights[coo][0][0]);
//                uint6 tmp01 = compute_engine_64(bottom0[row-1][col  ], weights[coo][0][1]);
//                uint6 tmp02 = compute_engine_64(bottom0[row-1][col+1], weights[coo][0][2]);
//                uint6 tmp03 = compute_engine_64(bottom0[row  ][col-1], weights[coo][1][0]);
//                uint6 tmp04 = compute_engine_64(bottom0[row  ][col  ], weights[coo][1][1]);
//                uint6 tmp05 = compute_engine_64(bottom0[row  ][col+1], weights[coo][1][2]);
//                uint6 tmp06 = compute_engine_64(bottom0[row+1][col-1], weights[coo][2][0]);
//                uint6 tmp07 = compute_engine_64(bottom0[row+1][col  ], weights[coo][2][1]);
//                uint6 tmp08 = compute_engine_64(bottom0[row+1][col+1], weights[coo][2][2]);
//                uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
//
//                if (sum > thres[coo]) {
//                    sum += sum0;
//                }
//							FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
//							d[w_i] += relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
							d[w_i] = sum;
							top[w_i][row][col] = d[w_i];
						}
					}
				}
				if (col != 7){
					for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
						d[i] = top[1*16 + i][row][col+1];
					}
				}
			}
		}
		if (row != 7){
			for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
				d[i] = top[0*16 + i][row+1][0];
			}
		}
    }
}

//#include "net_hls.h"
//#include <stdio.h>
//#include <math.h>
//#include <ap_fixed.h>
//#include "hls_stream.h"
//
//
//const static uint4 lut16[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
//
//inline uint6 compute_engine_64(uint64 b, uint64 w)
//{
//#pragma HLS PIPELINE
//    uint64 t = ~(b^w);
//    ap_int<4> add0, add1, add2, add3, add4, add5, add6, add7;
//    ap_int<5> add8, add9, adda, addb;
//    ap_int<6> addc, addd;
//
//    add0 = lut16[(int)t.range(3,  0 )] + lut16[(int)t.range(7,  4 )];
//    add1 = lut16[(int)t.range(11, 8 )] + lut16[(int)t.range(15, 12)];
//    add2 = lut16[(int)t.range(19, 16)] + lut16[(int)t.range(23, 20)];
//    add3 = lut16[(int)t.range(27, 24)] + lut16[(int)t.range(31, 28)];
//    add4 = lut16[(int)t.range(35, 32)] + lut16[(int)t.range(39, 36)];
//    add5 = lut16[(int)t.range(43, 40)] + lut16[(int)t.range(47, 44)];
//    add6 = lut16[(int)t.range(51, 48)] + lut16[(int)t.range(55, 52)];
//    add7 = lut16[(int)t.range(59, 56)] + lut16[(int)t.range(63, 60)];
//
//    add8 = add0 + add1;
//    add9 = add2 + add3;
//    adda = add4 + add5;
//    addb = add6 + add7;
//
//    addc = add8 + add9;
//    addd = adda + addb;
//
//    return addc + addd;
//}
//
//inline uint6 compute_engine_32(uint32 b, uint32 w)
//{
//#pragma HLS PIPELINE
//    uint32 t = ~(b^w);
//    ap_int<4> add0, add1, add2, add3;
//    ap_int<5> add4, add5;
//
//    add0 = lut16[(int)t.range(3,  0 )] + lut16[(int)t.range(7,  4 )];
//    add1 = lut16[(int)t.range(11, 8 )] + lut16[(int)t.range(15, 12)];
//    add2 = lut16[(int)t.range(19, 16)] + lut16[(int)t.range(23, 20)];
//    add3 = lut16[(int)t.range(27, 24)] + lut16[(int)t.range(31, 28)];
//
//    add4 = add0 + add1;
//    add5 = add2 + add3;
//
//    return add4 + add5;
//}
//
//inline uint8 sum_engine(uint6 t0,
//        uint6 t1,
//        uint6 t2,
//        uint6 t3,
//        uint6 t4,
//        uint6 t5,
//        uint6 t6,
//        uint6 t7,
//        uint6 t8)
//{
//#pragma HLS PIPELINE
//    ap_int<6> add0, add1, add2, add3;
//    ap_int<7> add4, add5, add6;
//
//    add0 = t0 + t1;
//    add1 = t2 + t3;
//    add2 = t4 + t5;
//    add3 = t6 + t7;
//
//    add4 = add0 + add1;
//    add5 = add2 + add3;
//
//    add6 = add4 + add5;
//
//    return add6 + t8;
//}
//
//inline FIX_FM_acc batch_norm(uint8 sum, FIX_WT weight, FIX_WT bias)
//{
//	FIX_FM_acc sw = sum*weight;
//    return sw + bias;
//}
//
//inline FIX_FM_acc relu(FIX_FM_acc norm, FIX_WT shiftx, FIX_WT shifty, FIX_WT weight)
//{
//    FIX_FM_acc tmp = norm + shiftx;
//    if (tmp > 0) {
//        return tmp + shifty;
//    } else {
//        return tmp*weight + shifty;
//    }
//}
//
//void pgconv32_2bit(uint2 bottom[32][32][32],
//                    uint32 weights[32][3][3],
//        			FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
//                    FIX_FM_acc top[32][32][32]
//)
//{
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//
//	for (int row = 1; row < 31; row ++) {
//		for (int col = 1; col < 31; col ++) {
//#pragma HLS PIPELINE
//			uint32 bottom_buf_1[3][3];
//			uint32 bottom_buf_0[3][3];
//			for (int cii = 0; cii < 32; cii ++) {
//#pragma HLS UNROLL
//				bottom_buf_1[0][0][cii] = bottom[cii][row-1][col-1][1];
//                bottom_buf_1[0][1][cii] = bottom[cii][row-1][col  ][1];
//                bottom_buf_1[0][2][cii] = bottom[cii][row-1][col+1][1];
//                bottom_buf_1[1][0][cii] = bottom[cii][row  ][col-1][1];
//                bottom_buf_1[1][1][cii] = bottom[cii][row  ][col  ][1];
//                bottom_buf_1[1][2][cii] = bottom[cii][row  ][col+1][1];
//                bottom_buf_1[2][0][cii] = bottom[cii][row+1][col-1][1];
//                bottom_buf_1[2][1][cii] = bottom[cii][row+1][col  ][1];
//                bottom_buf_1[2][2][cii] = bottom[cii][row+1][col+1][1];
//				bottom_buf_0[0][0][cii] = bottom[cii][row-1][col-1][0];
//                bottom_buf_0[0][1][cii] = bottom[cii][row-1][col  ][0];
//                bottom_buf_0[0][2][cii] = bottom[cii][row-1][col+1][0];
//                bottom_buf_0[1][0][cii] = bottom[cii][row  ][col-1][0];
//                bottom_buf_0[1][1][cii] = bottom[cii][row  ][col  ][0];
//                bottom_buf_0[1][2][cii] = bottom[cii][row  ][col+1][0];
//                bottom_buf_0[2][0][cii] = bottom[cii][row+1][col-1][0];
//                bottom_buf_0[2][1][cii] = bottom[cii][row+1][col  ][0];
//                bottom_buf_0[2][2][cii] = bottom[cii][row+1][col+1][0];
//			}
//			for (int coo = 0; coo < 32; coo ++) {
//#pragma HLS UNROLL
//				uint6 tmp0 = compute_engine_32(bottom_buf_1[row-1][col-1], weights[coo][0][0]);
//				uint6 tmp1 = compute_engine_32(bottom_buf_1[row-1][col  ], weights[coo][0][1]);
//				uint6 tmp2 = compute_engine_32(bottom_buf_1[row-1][col+1], weights[coo][0][2]);
//				uint6 tmp3 = compute_engine_32(bottom_buf_1[row  ][col-1], weights[coo][1][0]);
//				uint6 tmp4 = compute_engine_32(bottom_buf_1[row  ][col  ], weights[coo][1][1]);
//				uint6 tmp5 = compute_engine_32(bottom_buf_1[row  ][col+1], weights[coo][1][2]);
//				uint6 tmp6 = compute_engine_32(bottom_buf_1[row+1][col-1], weights[coo][2][0]);
//				uint6 tmp7 = compute_engine_32(bottom_buf_1[row+1][col  ], weights[coo][2][1]);
//				uint6 tmp8 = compute_engine_32(bottom_buf_1[row+1][col+1], weights[coo][2][2]);
//				uint8 sum = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
//
//				uint6 tmp00 = compute_engine_32(bottom_buf_0[row-1][col-1], weights[coo][0][0]);
//				uint6 tmp01 = compute_engine_32(bottom_buf_0[row-1][col  ], weights[coo][0][1]);
//				uint6 tmp02 = compute_engine_32(bottom_buf_0[row-1][col+1], weights[coo][0][2]);
//				uint6 tmp03 = compute_engine_32(bottom_buf_0[row  ][col-1], weights[coo][1][0]);
//				uint6 tmp04 = compute_engine_32(bottom_buf_0[row  ][col  ], weights[coo][1][1]);
//				uint6 tmp05 = compute_engine_32(bottom_buf_0[row  ][col+1], weights[coo][1][2]);
//				uint6 tmp06 = compute_engine_32(bottom_buf_0[row+1][col-1], weights[coo][2][0]);
//				uint6 tmp07 = compute_engine_32(bottom_buf_0[row+1][col  ], weights[coo][2][1]);
//				uint6 tmp08 = compute_engine_32(bottom_buf_0[row+1][col+1], weights[coo][2][2]);
//				uint8 sum0 = sum_engine(tmp00, tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08);
//
//				if (sum > thres[coo]) {
//					sum += sum0;
//				}
//				FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
//				top[coo][row][col] = relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//			}
//		}
//	}
//}
//
//void pgconv32_1bit(uint1 bottom[32][11][11],
//                    uint32 weights[32][3][3],
//        			FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
//                    FIX_FM_acc top[32][11][11],
//					uint1 stride
//)
//{
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//
//	for (int row = 2; row < 9; row +=2) {
//		for (int col = 2; col < 9; col +=2) {
//#pragma HLS PIPELINE II=5
//			uint32 bottom_buf_0[3][3];
//			for (int cii = 0; cii < 32; cii ++) {
//#pragma HLS UNROLL
//				bottom_buf_0[0][0][cii] = bottom[cii][row-1][col-1];
//                bottom_buf_0[0][1][cii] = bottom[cii][row-1][col  ];
//                bottom_buf_0[0][2][cii] = bottom[cii][row-1][col+1];
//                bottom_buf_0[1][0][cii] = bottom[cii][row  ][col-1];
//                bottom_buf_0[1][1][cii] = bottom[cii][row  ][col  ];
//                bottom_buf_0[1][2][cii] = bottom[cii][row  ][col+1];
//                bottom_buf_0[2][0][cii] = bottom[cii][row+1][col-1];
//                bottom_buf_0[2][1][cii] = bottom[cii][row+1][col  ];
//                bottom_buf_0[2][2][cii] = bottom[cii][row+1][col+1];
//			}
//			for (int coo = 0; coo < 32; coo ++) {
//#pragma HLS UNROLL
////				FIX_FM_acc d= top[coo][row][col];
//
//				uint6 tmp0 = compute_engine_32(bottom_buf_0[0][0], weights[coo][0][0]);
//				uint6 tmp1 = compute_engine_32(bottom_buf_0[0][1], weights[coo][0][1]);
//				uint6 tmp2 = compute_engine_32(bottom_buf_0[0][2], weights[coo][0][2]);
//				uint6 tmp3 = compute_engine_32(bottom_buf_0[1][0], weights[coo][1][0]);
//				uint6 tmp4 = compute_engine_32(bottom_buf_0[1][1], weights[coo][1][1]);
//				uint6 tmp5 = compute_engine_32(bottom_buf_0[1][2], weights[coo][1][2]);
//				uint6 tmp6 = compute_engine_32(bottom_buf_0[2][0], weights[coo][2][0]);
//				uint6 tmp7 = compute_engine_32(bottom_buf_0[2][1], weights[coo][2][1]);
//				uint6 tmp8 = compute_engine_32(bottom_buf_0[2][2], weights[coo][2][2]);
//				uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
//
//				FIX_FM_acc norm = batch_norm(sum0, bn_weights[coo], bn_bias[coo]);
//				top[coo][row][col] = relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//			}
//		}
//	}
//}
//
////void pgconv32_1bit(uint1 bottom[32][32][32],
////                    uint32 weights[32][3][3],
////        			FIX_WT thres[32],
////                    FIX_WT bn_weights[32],
////                    FIX_WT bn_bias[32],
////                    FIX_WT relu_shiftx[32],
////                    FIX_WT relu_shifty[32],
////                    FIX_WT relu_weights[32],
////                    FIX_FM_acc top[32][32][32]
////)
////{
////#pragma HLS array_partition variable=bottom dim=1 complete
////#pragma HLS array_partition variable=weights dim=1 complete
////#pragma HLS array_partition variable=thres dim=1 complete
////#pragma HLS array_partition variable=bn_weights dim=1 complete
////#pragma HLS array_partition variable=bn_bias dim=1 complete
////#pragma HLS array_partition variable=relu_shiftx dim=1 complete
////#pragma HLS array_partition variable=relu_shifty dim=1 complete
////#pragma HLS array_partition variable=relu_weights dim=1 complete
////#pragma HLS array_partition variable=top dim=1 complete
////
////	for (int row = 1; row < 31; row ++) {
////		for (int col = 1; col < 31; col ++) {
////#pragma HLS PIPELINE II=5
////			uint32 bottom_buf_0[3][3];
////			for (int cii = 0; cii < 32; cii ++) {
////#pragma HLS UNROLL
////				bottom_buf_0[0][0][cii] = bottom[cii][row-1][col-1];
////                bottom_buf_0[0][1][cii] = bottom[cii][row-1][col  ];
////                bottom_buf_0[0][2][cii] = bottom[cii][row-1][col+1];
////                bottom_buf_0[1][0][cii] = bottom[cii][row  ][col-1];
////                bottom_buf_0[1][1][cii] = bottom[cii][row  ][col  ];
////                bottom_buf_0[1][2][cii] = bottom[cii][row  ][col+1];
////                bottom_buf_0[2][0][cii] = bottom[cii][row+1][col-1];
////                bottom_buf_0[2][1][cii] = bottom[cii][row+1][col  ];
////                bottom_buf_0[2][2][cii] = bottom[cii][row+1][col+1];
////			}
////			for (int coo = 0; coo < 32; coo ++) {
////#pragma HLS UNROLL
////				uint6 tmp0 = compute_engine_32(bottom_buf_0[0][0], weights[coo][0][0]);
////				uint6 tmp1 = compute_engine_32(bottom_buf_0[0][1], weights[coo][0][1]);
////				uint6 tmp2 = compute_engine_32(bottom_buf_0[0][2], weights[coo][0][2]);
////				uint6 tmp3 = compute_engine_32(bottom_buf_0[1][0], weights[coo][1][0]);
////				uint6 tmp4 = compute_engine_32(bottom_buf_0[1][1], weights[coo][1][1]);
////				uint6 tmp5 = compute_engine_32(bottom_buf_0[1][2], weights[coo][1][2]);
////				uint6 tmp6 = compute_engine_32(bottom_buf_0[2][0], weights[coo][2][0]);
////				uint6 tmp7 = compute_engine_32(bottom_buf_0[2][1], weights[coo][2][1]);
////				uint6 tmp8 = compute_engine_32(bottom_buf_0[2][2], weights[coo][2][2]);
////				uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
////
////				FIX_FM_acc norm = batch_norm(sum0, bn_weights[coo], bn_bias[coo]);
////				top[coo][row][col] = relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
////			}
////		}
////	}
////}
//
//void pgconv32_1bit_alt(uint1 bottom[32][32][32],
//                    uint32 weights[32][3][3],
//        			FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
//                    FIX_FM_acc top[32][32][32]
//)
//{
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//
//	uint32 bottom_buf_0[32][32];
//	for (int row = 1; row < 31; row ++) {
//		for (int col = 1; col < 31; col ++) {
//#pragma HLS PIPELINE
//			for (int cii = 0; cii < 32; cii ++) {
//#pragma HLS UNROLL
//				bottom_buf_0[row][col][cii] = bottom[cii][row][col];
//			}
//		}
//	}
//	for (int row = 1; row < 31; row ++) {
//		for (int col = 1; col < 31; col ++) {
//#pragma HLS PIPELINE II=5
//			for (int coo = 0; coo < 32; coo ++) {
//#pragma HLS UNROLL
//				uint6 tmp00 = compute_engine_32(bottom_buf_0[row-1][col-1], weights[coo][0][0]);
//				uint6 tmp01 = compute_engine_32(bottom_buf_0[row-1][col  ], weights[coo][0][1]);
//				uint6 tmp02 = compute_engine_32(bottom_buf_0[row-1][col+1], weights[coo][0][2]);
//				uint6 tmp03 = compute_engine_32(bottom_buf_0[row  ][col-1], weights[coo][1][0]);
//				uint6 tmp04 = compute_engine_32(bottom_buf_0[row  ][col  ], weights[coo][1][1]);
//				uint6 tmp05 = compute_engine_32(bottom_buf_0[row  ][col+1], weights[coo][1][2]);
//				uint6 tmp06 = compute_engine_32(bottom_buf_0[row+1][col-1], weights[coo][2][0]);
//				uint6 tmp07 = compute_engine_32(bottom_buf_0[row+1][col  ], weights[coo][2][1]);
//				uint6 tmp08 = compute_engine_32(bottom_buf_0[row+1][col+1], weights[coo][2][2]);
//				uint8 sum0 = sum_engine(tmp00, tmp01, tmp02, tmp03, tmp04, tmp05, tmp06, tmp07, tmp08);
//
//				FIX_FM_acc norm = batch_norm(sum0, bn_weights[coo], bn_bias[coo]);
//				top[coo][row][col] = relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//			}
//		}
//	}
//}
//
//void pgconv32_1x1_2bit(uint2 bottom[32][32][32],
//                    uint32 weights[32],
//        			FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
//                    FIX_FM_acc top[32][32][32]
//)
//{
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//
//	for (int row = 1; row < 31; row ++) {
//		for (int col = 1; col < 31; col ++) {
//#pragma HLS PIPELINE II=5
//			uint32 bottom_buf_1;
//			uint32 bottom_buf_0;
//			for (int cii = 0; cii < 32; cii ++) {
//#pragma HLS UNROLL
//				bottom_buf_1[cii] = bottom[cii][row][col][1];
//				bottom_buf_0[cii] = bottom[cii][row][col][0];
//			}
//			for (int coo = 0; coo < 32; coo ++) {
//#pragma HLS UNROLL
//				uint6 tmp1 = compute_engine_32(bottom_buf_1, weights[coo]);
//				uint6 tmp0 = compute_engine_32(bottom_buf_0, weights[coo]);
//				uint8 sum = tmp1;
//				if (tmp1 > thres[coo]) {
//					sum += tmp0;
//				}
//				FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
//				top[coo][row][col] = relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//			}
//		}
//	}
//}
//
//void pgconv32_1x1_1bit(uint1 bottom[32][11][11],
//                    uint32 weights[32],
//                    FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
//                    FIX_FM_acc top[32][11][11]
//)
//{
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//
//    for (int row = 2; row < 9; row ++) {
//        for (int col = 2; col < 9; col ++) {
//#pragma HLS PIPELINE II=5
//            uint32 bottom_buf_1;
//            for (int cii = 0; cii < 32; cii ++) {
//#pragma HLS UNROLL
//                bottom_buf_1[cii] = bottom[cii][row][col];
//            }
//            for (int coo = 0; coo < 32; coo ++) {
//#pragma HLS UNROLL
//                uint6 tmp1 = compute_engine_32(bottom_buf_1, weights[coo]);
//                uint8 sum = tmp1;
//                FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
//                top[coo][row][col] = relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//            }
//        }
//    }
//}
//
////void pgconv32_1x1_1bit(uint1 bottom[32][32][32],
////                    uint32 weights[32],
////                    FIX_WT thres[32],
////                    FIX_WT bn_weights[32],
////                    FIX_WT bn_bias[32],
////                    FIX_WT relu_shiftx[32],
////                    FIX_WT relu_shifty[32],
////                    FIX_WT relu_weights[32],
////                    FIX_FM_acc top[32][32][32]
////)
////{
////#pragma HLS array_partition variable=bottom dim=1 complete
////#pragma HLS array_partition variable=weights dim=1 complete
////#pragma HLS array_partition variable=thres dim=1 complete
////#pragma HLS array_partition variable=bn_weights dim=1 complete
////#pragma HLS array_partition variable=bn_bias dim=1 complete
////#pragma HLS array_partition variable=relu_shiftx dim=1 complete
////#pragma HLS array_partition variable=relu_shifty dim=1 complete
////#pragma HLS array_partition variable=relu_weights dim=1 complete
////#pragma HLS array_partition variable=top dim=1 complete
////
////    for (int row = 1; row < 31; row ++) {
////        for (int col = 1; col < 31; col ++) {
////#pragma HLS PIPELINE II=5
////            uint32 bottom_buf_1;
////            for (int cii = 0; cii < 32; cii ++) {
////#pragma HLS UNROLL
////                bottom_buf_1[cii] = bottom[cii][row][col];
////            }
////            for (int coo = 0; coo < 32; coo ++) {
////#pragma HLS UNROLL
////                uint6 tmp1 = compute_engine_32(bottom_buf_1, weights[coo]);
////                uint8 sum = tmp1;
////                FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
////                top[coo][row][col] = relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
////            }
////        }
////    }
////}
//void pgconv64_1bit(uint64 bottom[9][9],
//                    uint64 weights[32][3][3],
//                    FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
//                    FIX_FM_acc top[32][9][9],
//                    int stride
//)
//{
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//
//    int s;
//    if (stride == 1) {
//    	s = 7;
//    } else {
//        s = 4;
//    }
//    for (int row0 = 0; row0 < s; row0 ++) {
//        for (int col0 = 0; col0 < s; col0 ++) {
//#pragma HLS PIPELINE II=5
//            int row, col;
//            if (stride == 1) {
//                row = row0 + 1;
//                col = col0 + 1;
//            } else {
//                row = row0*2 + 1;
//                col = col0*2 + 1;
//            }
//            uint64 bottom_buf_0[3][3];
////            for (int cii = 0; cii < 64; cii ++) {
////#pragma HLS UNROLL
////                bottom_buf_0[0][0][cii] = bottom[cii][row-1][col-1];
////                bottom_buf_0[0][1][cii] = bottom[cii][row-1][col  ];
////                bottom_buf_0[0][2][cii] = bottom[cii][row-1][col+1];
////                bottom_buf_0[1][0][cii] = bottom[cii][row  ][col-1];
////                bottom_buf_0[1][1][cii] = bottom[cii][row  ][col  ];
////                bottom_buf_0[1][2][cii] = bottom[cii][row  ][col+1];
////                bottom_buf_0[2][0][cii] = bottom[cii][row+1][col-1];
////                bottom_buf_0[2][1][cii] = bottom[cii][row+1][col  ];
////                bottom_buf_0[2][2][cii] = bottom[cii][row+1][col+1];
////            }
//            for (int coo = 0; coo < 32; coo ++) {
//#pragma HLS UNROLL
//                FIX_FM_acc d = top[coo][row][col];
//                uint6 tmp0 = compute_engine_64(bottom[row-1][col-1], weights[coo][0][0]);
//                uint6 tmp1 = compute_engine_64(bottom[row-1][col  ], weights[coo][0][1]);
//                uint6 tmp2 = compute_engine_64(bottom[row-1][col+1], weights[coo][0][2]);
//                uint6 tmp3 = compute_engine_64(bottom[row  ][col-1], weights[coo][1][0]);
//                uint6 tmp4 = compute_engine_64(bottom[row  ][col  ], weights[coo][1][1]);
//                uint6 tmp5 = compute_engine_64(bottom[row  ][col+1], weights[coo][1][2]);
//                uint6 tmp6 = compute_engine_64(bottom[row+1][col-1], weights[coo][2][0]);
//                uint6 tmp7 = compute_engine_64(bottom[row+1][col  ], weights[coo][2][1]);
//                uint6 tmp8 = compute_engine_64(bottom[row+1][col+1], weights[coo][2][2]);
//                uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
//
//                FIX_FM_acc norm = batch_norm(sum0, bn_weights[coo], bn_bias[coo]);
//                top[coo][row][col] = d + relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//            }
//        }
//    }
//}
//
//void pgconv64_1bit_s2(uint1 bottom[32][11][11],
//                    uint64 weights[32][3][3],
//                    FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
//                    FIX_FM_acc top[32][11][11],
//                    int stride
//)
//{
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//
//    for (int row = 2; row < 9; row +=2) {
//        for (int col = 2; col < 9; col +=2) {
//#pragma HLS PIPELINE II=5
//            uint64 bottom_buf_0[3][3];
//            for (int cii = 0; cii < 32; cii ++) {
//#pragma HLS UNROLL
//                bottom_buf_0[0][0][cii] = bottom[cii][row-1][col-1];
//                bottom_buf_0[0][1][cii] = bottom[cii][row-1][col  ];
//                bottom_buf_0[0][2][cii] = bottom[cii][row-1][col+1];
//                bottom_buf_0[1][0][cii] = bottom[cii][row  ][col-1];
//                bottom_buf_0[1][1][cii] = bottom[cii][row  ][col  ];
//                bottom_buf_0[1][2][cii] = bottom[cii][row  ][col+1];
//                bottom_buf_0[2][0][cii] = bottom[cii][row+1][col-1];
//                bottom_buf_0[2][1][cii] = bottom[cii][row+1][col  ];
//                bottom_buf_0[2][2][cii] = bottom[cii][row+1][col+1];
//            }
//            for (int coo = 0; coo < 32; coo ++) {
//#pragma HLS UNROLL
//            	FIX_FM_acc d = top[coo][row][col];
//                uint6 tmp0 = compute_engine_64(bottom_buf_0[0][0], weights[coo][0][0]);
//                uint6 tmp1 = compute_engine_64(bottom_buf_0[0][1], weights[coo][0][1]);
//                uint6 tmp2 = compute_engine_64(bottom_buf_0[0][2], weights[coo][0][2]);
//                uint6 tmp3 = compute_engine_64(bottom_buf_0[1][0], weights[coo][1][0]);
//                uint6 tmp4 = compute_engine_64(bottom_buf_0[1][1], weights[coo][1][1]);
//                uint6 tmp5 = compute_engine_64(bottom_buf_0[1][2], weights[coo][1][2]);
//                uint6 tmp6 = compute_engine_64(bottom_buf_0[2][0], weights[coo][2][0]);
//                uint6 tmp7 = compute_engine_64(bottom_buf_0[2][1], weights[coo][2][1]);
//                uint6 tmp8 = compute_engine_64(bottom_buf_0[2][2], weights[coo][2][2]);
//                uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
//
//                FIX_FM_acc norm = batch_norm(sum0, bn_weights[coo], bn_bias[coo]);
//                top[coo][row][col] = d + relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//            }
//        }
//    }
//}
//
//void pgconv64_2bit(uint1 bottom1[32][11][11],
//					uint1 bottom0[32][11][11],
//                    uint64 weights[32][3][3],
//                    FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
//                    FIX_FM_acc top[32][11][11],
//                    int stride
//)
//{
//#pragma HLS array_partition variable=bottom1 dim=1 complete
//#pragma HLS array_partition variable=bottom0 dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//
//    int s = 9;
////  if (stride == 1) {
////      s = 9;
////  } else {
////	s = 5;
////  }
//    for (int row = 2; row < s; row ++) {
//        for (int col = 2; col < s; col ++) {
//#pragma HLS PIPELINE II=5
//            uint64 bottom_buf_1[3][3];
//            uint64 bottom_buf_0[3][3];
//            for (int cii = 0; cii < 32; cii ++) {
//#pragma HLS UNROLL
//                bottom_buf_1[0][0][cii] = bottom1[cii][row-1][col-1];
//                bottom_buf_1[0][1][cii] = bottom1[cii][row-1][col  ];
//                bottom_buf_1[0][2][cii] = bottom1[cii][row-1][col+1];
//                bottom_buf_1[1][0][cii] = bottom1[cii][row  ][col-1];
//                bottom_buf_1[1][1][cii] = bottom1[cii][row  ][col  ];
//                bottom_buf_1[1][2][cii] = bottom1[cii][row  ][col+1];
//                bottom_buf_1[2][0][cii] = bottom1[cii][row+1][col-1];
//                bottom_buf_1[2][1][cii] = bottom1[cii][row+1][col  ];
//                bottom_buf_1[2][2][cii] = bottom1[cii][row+1][col+1];
//                bottom_buf_0[0][0][cii] = bottom0[cii][row-1][col-1];
//                bottom_buf_0[0][1][cii] = bottom0[cii][row-1][col  ];
//                bottom_buf_0[0][2][cii] = bottom0[cii][row-1][col+1];
//                bottom_buf_0[1][0][cii] = bottom0[cii][row  ][col-1];
//                bottom_buf_0[1][1][cii] = bottom0[cii][row  ][col  ];
//                bottom_buf_0[1][2][cii] = bottom0[cii][row  ][col+1];
//                bottom_buf_0[2][0][cii] = bottom0[cii][row+1][col-1];
//                bottom_buf_0[2][1][cii] = bottom0[cii][row+1][col  ];
//                bottom_buf_0[2][2][cii] = bottom0[cii][row+1][col+1];
//            }
//            for (int coo = 0; coo < 32; coo ++) {
//#pragma HLS UNROLL
//            	FIX_FM_acc d = top[coo][row][col];
//                uint6 tmp0 = compute_engine_64(bottom_buf_1[0][0], weights[coo][0][0]);
//                uint6 tmp1 = compute_engine_64(bottom_buf_1[0][1], weights[coo][0][1]);
//                uint6 tmp2 = compute_engine_64(bottom_buf_1[0][2], weights[coo][0][2]);
//                uint6 tmp3 = compute_engine_64(bottom_buf_1[1][0], weights[coo][1][0]);
//                uint6 tmp4 = compute_engine_64(bottom_buf_1[1][1], weights[coo][1][1]);
//                uint6 tmp5 = compute_engine_64(bottom_buf_1[1][2], weights[coo][1][2]);
//                uint6 tmp6 = compute_engine_64(bottom_buf_1[2][0], weights[coo][2][0]);
//                uint6 tmp7 = compute_engine_64(bottom_buf_1[2][1], weights[coo][2][1]);
//                uint6 tmp8 = compute_engine_64(bottom_buf_1[2][2], weights[coo][2][2]);
//                uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
//
//                uint6 tmp10 = compute_engine_64(bottom_buf_0[0][0], weights[coo][0][0]);
//                uint6 tmp11 = compute_engine_64(bottom_buf_0[0][1], weights[coo][0][1]);
//                uint6 tmp12 = compute_engine_64(bottom_buf_0[0][2], weights[coo][0][2]);
//                uint6 tmp13 = compute_engine_64(bottom_buf_0[1][0], weights[coo][1][0]);
//                uint6 tmp14 = compute_engine_64(bottom_buf_0[1][1], weights[coo][1][1]);
//                uint6 tmp15 = compute_engine_64(bottom_buf_0[1][2], weights[coo][1][2]);
//                uint6 tmp16 = compute_engine_64(bottom_buf_0[2][0], weights[coo][2][0]);
//                uint6 tmp17 = compute_engine_64(bottom_buf_0[2][1], weights[coo][2][1]);
//                uint6 tmp18 = compute_engine_64(bottom_buf_0[2][2], weights[coo][2][2]);
//                uint8 sum10 = sum_engine(tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17, tmp18);
//
//                if (sum0 > thres[coo]) {
//                    sum0 += sum10;
//                }
//
//                FIX_FM_acc norm = batch_norm(sum0, bn_weights[coo], bn_bias[coo]);
//                top[coo][row][col] = d + relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//            }
//        }
//    }
//}
//
//void pgconv64_1x1_1bit(uint64 bottom[9][9],
//                    uint64 weights[32],
//                    FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
//                    FIX_FM_acc top[32][9][9]
//)
//{
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//
//    for (int row = 1; row < 8; row ++) {
//        for (int col = 1; col < 8; col ++) {
//#pragma HLS PIPELINE II=5
////            uint64 bottom_buf_1;
////            for (int cii = 0; cii < 64; cii ++) {
////#pragma HLS UNROLL
////                bottom_buf_1[cii] = bottom[cii][row][col];
////            }
//            for (int coo = 0; coo < 32; coo ++) {
//#pragma HLS UNROLL
//                FIX_FM_acc d = top[coo][row][col];
//                uint6 tmp1 = compute_engine_64(bottom[row][col], weights[coo]);
//                uint8 sum = tmp1;
//                FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
//                top[coo][row][col] = d + relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//            }
//        }
//    }
//}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
////
////#include "net_hls.h"
////#include <stdio.h>
////#include <math.h>
////#include <ap_fixed.h>
////#include "hls_stream.h"
////
////
////const static uint4 lut16[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
////
////uint6 compute_engine_64(uint64 b, uint64 w)
////{
////#pragma HLS PIPELINE
////    uint64 t = ~(b^w);
////    ap_int<4> add0, add1, add2, add3, add4, add5, add6, add7;
////    ap_int<5> add8, add9, adda, addb;
////    ap_int<6> addc, addd;
////
////    add0 = lut16[(int)t.range(3,  0 )] + lut16[(int)t.range(7,  4 )];
////    add1 = lut16[(int)t.range(11, 8 )] + lut16[(int)t.range(15, 12)];
////    add2 = lut16[(int)t.range(19, 16)] + lut16[(int)t.range(23, 20)];
////    add3 = lut16[(int)t.range(27, 24)] + lut16[(int)t.range(31, 28)];
////    add4 = lut16[(int)t.range(35, 32)] + lut16[(int)t.range(39, 36)];
////    add5 = lut16[(int)t.range(43, 40)] + lut16[(int)t.range(47, 44)];
////    add6 = lut16[(int)t.range(51, 48)] + lut16[(int)t.range(55, 52)];
////    add7 = lut16[(int)t.range(59, 56)] + lut16[(int)t.range(63, 60)];
////
////    add8 = add0 + add1;
////    add9 = add2 + add3;
////    adda = add4 + add5;
////    addb = add6 + add7;
////
////    addc = add8 + add9;
////    addd = adda + addb;
////
////    return addc + addd;
////}
////
////inline uint6 compute_engine_32(uint32 b, uint32 w)
////{
////#pragma HLS PIPELINE
////    uint32 t = ~(b^w);
////    ap_int<4> add0, add1, add2, add3;
////    ap_int<5> add4, add5;
////
////    add0 = lut16[(int)t.range(3,  0 )] + lut16[(int)t.range(7,  4 )];
////    add1 = lut16[(int)t.range(11, 8 )] + lut16[(int)t.range(15, 12)];
////    add2 = lut16[(int)t.range(19, 16)] + lut16[(int)t.range(23, 20)];
////    add3 = lut16[(int)t.range(27, 24)] + lut16[(int)t.range(31, 28)];
////
////    add4 = add0 + add1;
////    add5 = add2 + add3;
////
////    return add4 + add5;
////}
////
////inline uint8 sum_engine(uint6 t0,
////        uint6 t1,
////        uint6 t2,
////        uint6 t3,
////        uint6 t4,
////        uint6 t5,
////        uint6 t6,
////        uint6 t7,
////        uint6 t8)
////{
////#pragma HLS PIPELINE
////    ap_int<6> add0, add1, add2, add3;
////    ap_int<7> add4, add5, add6;
////
////    add0 = t0 + t1;
////    add1 = t2 + t3;
////    add2 = t4 + t5;
////    add3 = t6 + t7;
////
////    add4 = add0 + add1;
////    add5 = add2 + add3;
////
////    add6 = add4 + add5;
////
////    return add6 + t8;
////}
////
////inline FIX_FM_acc batch_norm(uint8 sum, FIX_WT weight, FIX_WT bias)
////{
////	FIX_FM_acc sw = sum*weight;
////    return sw + bias;
////}
////
////inline FIX_FM_acc relu(FIX_FM_acc norm, FIX_WT shiftx, FIX_WT shifty, FIX_WT weight)
////{
////    FIX_FM_acc tmp = norm + shiftx;
////    if (tmp > 0) {
////        return tmp + shifty;
////    } else {
////        return tmp*weight + shifty;
////    }
////}
////
////void pgconv64_1bit(uint64 bottom[9][9],
////                    uint64 weights[32][3][3],
////                    FIX_WT thres[32],
////                    FIX_WT bn_weights[32],
////                    FIX_WT bn_bias[32],
////                    FIX_WT relu_shiftx[32],
////                    FIX_WT relu_shifty[32],
////                    FIX_WT relu_weights[32],
////                    FIX_FM_acc top[32][9][9],
////                    int stride
////)
////{
//////#pragma HLS ALLOCATION instances=compute_engine_64 limit=32 function
////
////#pragma HLS array_partition variable=bottom dim=1 complete
////#pragma HLS array_partition variable=weights dim=0 complete
////#pragma HLS array_partition variable=thres dim=1 complete
////#pragma HLS array_partition variable=bn_weights dim=1 complete
////#pragma HLS array_partition variable=bn_bias dim=1 complete
////#pragma HLS array_partition variable=relu_shiftx dim=1 complete
////#pragma HLS array_partition variable=relu_shifty dim=1 complete
////#pragma HLS array_partition variable=relu_weights dim=1 complete
////#pragma HLS array_partition variable=top dim=1 complete
//////    uint64 bot1_LB[3][10];
//////    uint64 bot1_WB[3][3];
//////    biconv_row:for(int row = 0; row < 9; row ++){ // 1-8
//////        biconv_col:for(int col = 0; col < 9; col ++) { // 1-8
//////#pragma HLS PIPELINE ii=5
//////            // move up one row, update line buffer
//////            bot1_LB[0][col] = bot1_LB[1][col];
//////            bot1_LB[1][col] = bot1_LB[2][col];
//////            bot1_LB[2][col] = bottom[row][col];
//////            if (0 < row && row < 8) { // move left one column, update window buffer
//////                for (int LB_1 = 0; LB_1 < 3; ++LB_1) {
//////#pragma HLS UNROLL
//////                    bot1_WB[LB_1][0] = bot1_WB[LB_1][1];
//////                    bot1_WB[LB_1][1] = bot1_WB[LB_1][2];
//////                    bot1_WB[LB_1][2] = bot1_LB[LB_1][col];
//////                }
//////                if (0 < col && col < 8) {
//////                    biconv_coo:for (int coo = 0; coo < 32; coo ++) { // calculate all the channels
//////                    #pragma HLS UNROLL
//////                        int w_i = coo;
//////                        uint6 tmp0 = compute_engine_64(bot1_WB[0][0], weights[w_i][0][0]);
//////                        uint6 tmp1 = compute_engine_64(bot1_WB[0][1], weights[w_i][0][1]);
//////                        uint6 tmp2 = compute_engine_64(bot1_WB[0][2], weights[w_i][0][2]);
//////                        uint6 tmp3 = compute_engine_64(bot1_WB[1][0], weights[w_i][1][0]);
//////                        uint6 tmp4 = compute_engine_64(bot1_WB[1][1], weights[w_i][1][1]);
//////                        uint6 tmp5 = compute_engine_64(bot1_WB[1][2], weights[w_i][1][2]);
//////                        uint6 tmp6 = compute_engine_64(bot1_WB[2][0], weights[w_i][2][0]);
//////                        uint6 tmp7 = compute_engine_64(bot1_WB[2][1], weights[w_i][2][1]);
//////                        uint6 tmp8 = compute_engine_64(bot1_WB[2][2], weights[w_i][2][2]);
//////                        uint8 sum = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
//////
//////                        if (sum > thres[coo]) {
//////                            sum += 0;
//////                        }
//////                        FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
//////                        top[coo][row][col] += relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
//////                    }
//////                }
//////            }
//////        }
//////    }
////
////
////     for (int row = 0; row < 9; row ++) {
////         for (int col = 0; col < 9; col ++) {
//// #pragma HLS PIPELINE II=5
////			 for (int coo = 0; coo < 32; coo ++) {
////#pragma HLS UNROLL
////				 FIX_FM_acc d = top[coo][row][col];
////				 uint6 tmp0 = compute_engine_64(bottom[row-1][col-1], weights[coo][0][0]);
////				 uint6 tmp1 = compute_engine_64(bottom[row-1][col  ], weights[coo][0][1]);
////				 uint6 tmp2 = compute_engine_64(bottom[row-1][col+1], weights[coo][0][2]);
////				 uint6 tmp3 = compute_engine_64(bottom[row  ][col-1], weights[coo][1][0]);
////				 uint6 tmp4 = compute_engine_64(bottom[row  ][col  ], weights[coo][1][1]);
////				 uint6 tmp5 = compute_engine_64(bottom[row  ][col+1], weights[coo][1][2]);
////				 uint6 tmp6 = compute_engine_64(bottom[row+1][col-1], weights[coo][2][0]);
////				 uint6 tmp7 = compute_engine_64(bottom[row+1][col  ], weights[coo][2][1]);
////				 uint6 tmp8 = compute_engine_64(bottom[row+1][col+1], weights[coo][2][2]);
////				 uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
////
////				 FIX_FM_acc norm = batch_norm(sum0, bn_weights[coo], bn_bias[coo]);
////				 top[coo][row][col] = d + relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
////			 }
////         }
////     }
////}
////
void pgconv64_1x1_1bit(uint64 bottom[9][9],
                    uint64 weights[32],
                    FIX_WT thres[32],
//                    FIX_WT bn_weights[32],
//                    FIX_WT bn_bias[32],
//                    FIX_WT relu_shiftx[32],
//                    FIX_WT relu_shifty[32],
//                    FIX_WT relu_weights[32],
                    FIX_FM_acc top[32][9][9]
)
{
#pragma HLS array_partition variable=bottom dim=1 complete
#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=thres dim=1 complete
//#pragma HLS array_partition variable=bn_weights dim=1 complete
//#pragma HLS array_partition variable=bn_bias dim=1 complete
//#pragma HLS array_partition variable=relu_shiftx dim=1 complete
//#pragma HLS array_partition variable=relu_shifty dim=1 complete
//#pragma HLS array_partition variable=relu_weights dim=1 complete
#pragma HLS array_partition variable=top dim=1 complete

    uint64 bot1_LB[9];
    uint64 bot1_WB;
    for (int row = 0; row < 9; row ++) {
        for (int col = 0; col < 9; col ++) {
#pragma HLS PIPELINE
            bot1_LB[col] = bottom[row][col];
            if (0 < row && row < 9) { // move left one column, update window buffer
                bot1_WB = bot1_LB[col];
                if (0 < col && col < 9) {
                    for (int coo = 0; coo < 32; coo ++) {
#pragma HLS UNROLL
                        FIX_FM_acc d = top[coo][row][col];
                        uint6 tmp1 = compute_engine_64(bot1_WB, weights[coo]);
                        uint8 sum = tmp1;
//                        FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
//                        top[coo][row][col] = d + relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
                        top[coo][row][col] = d + sum;
                    }
                }
            }
        }
    }
}

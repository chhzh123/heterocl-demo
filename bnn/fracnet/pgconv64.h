#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>

const static uint4 lut16[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

inline uint6 compute_engine_64(uint64 b, uint64 w)
{
#pragma HLS PIPELINE
    uint64 t = ~(b^w);
    ap_int<4> add0, add1, add2, add3, add4, add5, add6, add7;
    ap_int<5> add8, add9, adda, addb;
    ap_int<6> addc, addd;

    add0 = lut16[(int)t.range(3,  0 )] + lut16[(int)t.range(7,  4 )];
    add1 = lut16[(int)t.range(11, 8 )] + lut16[(int)t.range(15, 12)];
    add2 = lut16[(int)t.range(19, 16)] + lut16[(int)t.range(23, 20)];
    add3 = lut16[(int)t.range(27, 24)] + lut16[(int)t.range(31, 28)];
    add4 = lut16[(int)t.range(35, 32)] + lut16[(int)t.range(39, 36)];
    add5 = lut16[(int)t.range(43, 40)] + lut16[(int)t.range(47, 44)];
    add6 = lut16[(int)t.range(51, 48)] + lut16[(int)t.range(55, 52)];
    add7 = lut16[(int)t.range(59, 56)] + lut16[(int)t.range(63, 60)];

    add8 = add0 + add1;
    add9 = add2 + add3;
    adda = add4 + add5;
    addb = add6 + add7;

    addc = add8 + add9;
    addd = adda + addb;

    return addc + addd;
}

inline FIX_FM_acc batch_norm(uint8 sum, FIX_WT weight, FIX_WT bias)
{
    return sum*weight + bias;
}

inline FIX_FM_acc relu(FIX_FM_acc norm, FIX_WT shiftx, FIX_WT shifty, FIX_WT weight)
{
    FIX_FM_acc tmp = norm + shiftx;
    if (tmp > 0) {
        return tmp + shifty;
    } else {
        return tmp*weight + shifty;
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

template <unsigned CHANNEL>
void pgconv64(uint64 bottom1[10][10],
				uint64 bottom0[10][10],
				int c,
				const uint64 weights[CHANNEL][3][3],
				const FIX_WT thres[CHANNEL],
				const FIX_WT bn_weights[CHANNEL],
				const FIX_WT bn_bias[CHANNEL],
				const FIX_WT relu_shiftx[CHANNEL],
				const FIX_WT relu_shifty[CHANNEL],
				const FIX_WT relu_weights[CHANNEL],
				FIX_FM_acc top[16][10][10]
)
{

#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=thres dim=1 complete
#pragma HLS array_partition variable=bn_weights dim=1 complete
#pragma HLS array_partition variable=bn_bias dim=1 complete
#pragma HLS array_partition variable=relu_shiftx dim=1 complete
#pragma HLS array_partition variable=relu_shifty dim=1 complete
#pragma HLS array_partition variable=relu_weights dim=1 complete
#pragma HLS array_partition variable=top dim=1 complete

    biconv_row:for(int row = 1; row < 9; row ++){
        biconv_col:for(int col = 1; col < 9; col ++) {
#pragma HLS PIPELINE II=5
            biconv_coo:for (int coo = 0; coo < 16; coo ++) {
#pragma HLS UNROLL
            	int w_i = c*16+coo;
                uint6 tmp0 = compute_engine_64(bottom1[row-1][col-1], weights[w_i][0][0]);
                uint6 tmp1 = compute_engine_64(bottom1[row-1][col  ], weights[w_i][0][1]);
                uint6 tmp2 = compute_engine_64(bottom1[row-1][col+1], weights[w_i][0][2]);
                uint6 tmp3 = compute_engine_64(bottom1[row  ][col-1], weights[w_i][1][0]);
                uint6 tmp4 = compute_engine_64(bottom1[row  ][col  ], weights[w_i][1][1]);
                uint6 tmp5 = compute_engine_64(bottom1[row  ][col+1], weights[w_i][1][2]);
                uint6 tmp6 = compute_engine_64(bottom1[row+1][col-1], weights[w_i][2][0]);
                uint6 tmp7 = compute_engine_64(bottom1[row+1][col  ], weights[w_i][2][1]);
                uint6 tmp8 = compute_engine_64(bottom1[row+1][col+1], weights[w_i][2][2]);
                uint8 sum = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);

                uint6 tmp00 = compute_engine_64(bottom0[row-1][col-1], weights[w_i][0][0]);
                uint6 tmp01 = compute_engine_64(bottom0[row-1][col  ], weights[w_i][0][1]);
                uint6 tmp02 = compute_engine_64(bottom0[row-1][col+1], weights[w_i][0][2]);
                uint6 tmp03 = compute_engine_64(bottom0[row  ][col-1], weights[w_i][1][0]);
                uint6 tmp04 = compute_engine_64(bottom0[row  ][col  ], weights[w_i][1][1]);
                uint6 tmp05 = compute_engine_64(bottom0[row  ][col+1], weights[w_i][1][2]);
                uint6 tmp06 = compute_engine_64(bottom0[row+1][col-1], weights[w_i][2][0]);
                uint6 tmp07 = compute_engine_64(bottom0[row+1][col  ], weights[w_i][2][1]);
                uint6 tmp08 = compute_engine_64(bottom0[row+1][col+1], weights[w_i][2][2]);
                uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);

                if (sum > thres[coo]) {
                    sum += sum0;
                }
                FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
                top[coo][row][col] = relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);

            }
        }
    }
}


template <unsigned CHANNEL>
void pgconv64s2(uint64 bottom1[10][10],
                uint64 bottom0[10][10],
                int c,
                int row_off,
                int col_off,
                const uint64 weights[CHANNEL][3][3],
                const FIX_WT thres[CHANNEL],
                const FIX_WT bn_weights[CHANNEL],
                const FIX_WT bn_bias[CHANNEL],
                const FIX_WT relu_shiftx[CHANNEL],
                const FIX_WT relu_shifty[CHANNEL],
                const FIX_WT relu_weights[CHANNEL],
                FIX_FM_acc top[16][10][10]
)
{

#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=thres dim=1 complete
#pragma HLS array_partition variable=bn_weights dim=1 complete
#pragma HLS array_partition variable=bn_bias dim=1 complete
#pragma HLS array_partition variable=relu_shiftx dim=1 complete
#pragma HLS array_partition variable=relu_shifty dim=1 complete
#pragma HLS array_partition variable=relu_weights dim=1 complete
#pragma HLS array_partition variable=top dim=1 complete

    int top_row = row_off*4 + 1;
    int top_col = col_off*4 + 1;
    biconv_row:for(int row = 1; row < 9; row +=2){
        biconv_col:for(int col = 1; col < 9; col +=2) {
#pragma HLS PIPELINE II=5
            biconv_coo:for (int coo = 0; coo < 16; coo ++) {
#pragma HLS UNROLL
                FIX_FM_acc d = top[coo][top_row][top_col];
                int w_i = c*16+coo;
                uint6 tmp0 = compute_engine_64(bottom1[row-1][col-1], weights[w_i][0][0]);
                uint6 tmp1 = compute_engine_64(bottom1[row-1][col  ], weights[w_i][0][1]);
                uint6 tmp2 = compute_engine_64(bottom1[row-1][col+1], weights[w_i][0][2]);
                uint6 tmp3 = compute_engine_64(bottom1[row  ][col-1], weights[w_i][1][0]);
                uint6 tmp4 = compute_engine_64(bottom1[row  ][col  ], weights[w_i][1][1]);
                uint6 tmp5 = compute_engine_64(bottom1[row  ][col+1], weights[w_i][1][2]);
                uint6 tmp6 = compute_engine_64(bottom1[row+1][col-1], weights[w_i][2][0]);
                uint6 tmp7 = compute_engine_64(bottom1[row+1][col  ], weights[w_i][2][1]);
                uint6 tmp8 = compute_engine_64(bottom1[row+1][col+1], weights[w_i][2][2]);
                uint8 sum = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);

                uint6 tmp00 = compute_engine_64(bottom0[row-1][col-1], weights[w_i][0][0]);
                uint6 tmp01 = compute_engine_64(bottom0[row-1][col  ], weights[w_i][0][1]);
                uint6 tmp02 = compute_engine_64(bottom0[row-1][col+1], weights[w_i][0][2]);
                uint6 tmp03 = compute_engine_64(bottom0[row  ][col-1], weights[w_i][1][0]);
                uint6 tmp04 = compute_engine_64(bottom0[row  ][col  ], weights[w_i][1][1]);
                uint6 tmp05 = compute_engine_64(bottom0[row  ][col+1], weights[w_i][1][2]);
                uint6 tmp06 = compute_engine_64(bottom0[row+1][col-1], weights[w_i][2][0]);
                uint6 tmp07 = compute_engine_64(bottom0[row+1][col  ], weights[w_i][2][1]);
                uint6 tmp08 = compute_engine_64(bottom0[row+1][col+1], weights[w_i][2][2]);
                uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);

                if (sum > thres[coo]) {
                    sum += sum0;
                }
                FIX_FM_acc norm = batch_norm(sum, bn_weights[coo], bn_bias[coo]);
                top[coo][top_row][top_col] = d + relu(norm, relu_shiftx[coo], relu_shifty[coo], relu_weights[coo]);
            }
            top_row ++;
            top_col ++;
        }
    }
}

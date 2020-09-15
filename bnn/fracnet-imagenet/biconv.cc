// Conv 1x1 PE

#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"

const static uint4 lut16[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

inline uint6 compute_engine_16(uint16 b, uint16 w)
{
#pragma HLS PIPELINE
    uint16 t = ~(b^w);
    ap_int<4> add0, add1;
    ap_int<5> add3;

    add0 = lut16[(int)t.range(3, 0)] + lut16[(int)t.range(7, 4)];
    add1 = lut16[(int)t.range(11, 8)] + lut16[(int)t.range(15, 12)];

    add3 = add0 + add1;

    return add3;
}

inline uint6 compute_engine_32(uint32 b, uint32 w)
{
#pragma HLS PIPELINE
    uint32 t = ~(b^w);
    ap_int<4> add0, add1, add2, add3;
    ap_int<5> add4, add5;

    add0 = lut16[(int)t.range(3,  0 )] + lut16[(int)t.range(7,  4 )];
    add1 = lut16[(int)t.range(11, 8 )] + lut16[(int)t.range(15, 12)];
    add2 = lut16[(int)t.range(19, 16)] + lut16[(int)t.range(23, 20)];
    add3 = lut16[(int)t.range(27, 24)] + lut16[(int)t.range(31, 28)];

    add4 = add0 + add1;
    add5 = add2 + add3;

    return add4 + add5;
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

inline FIX_FM_acc batch_norm(uint8 sum, FIX_WT weight, FIX_WT bias)
{
	FIX_WT wt = weight;
	FIX_WT bs = bias;
	FIX_FM_acc sw = sum*wt;
	FIX_FM_acc ret = sw + bs;
    return ret;
}

void biconv16(uint16 bottom[9][9],
        uint16 weights[32][3][3],
        FIX_WT bn_weight[32],
        FIX_WT bn_bias[32],
              FIX_FM_acc top[32][9][9]
)
{

#pragma HLS array_partition variable=top dim=1 complete
#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=bn_weight dim=1 complete
#pragma HLS array_partition variable=bn_bias dim=1 complete

	biconv_row:for(int row0 = 0; row0 < 4; row0 ++){
		biconv_col:for(int col0 = 0; col0 < 4; col0 ++) {
#pragma HLS PIPELINE II=5
			biconv_coo:for (int coo = 0; coo < 32; coo ++) {
#pragma HLS UNROLL
				FIX_FM_acc d = top[coo][row0][col0];
				int row = row0*2 + 2;
				int col = col0*2 + 2;
				uint6 tmp0 = compute_engine_16(bottom[row-1][col-1], weights[coo][0][0]);
				uint6 tmp1 = compute_engine_16(bottom[row-1][col  ], weights[coo][0][1]);
				uint6 tmp2 = compute_engine_16(bottom[row-1][col+1], weights[coo][0][2]);
				uint6 tmp3 = compute_engine_16(bottom[row  ][col-1], weights[coo][1][0]);
				uint6 tmp4 = compute_engine_16(bottom[row  ][col  ], weights[coo][1][1]);
				uint6 tmp5 = compute_engine_16(bottom[row  ][col+1], weights[coo][1][2]);
				uint6 tmp6 = compute_engine_16(bottom[row+1][col-1], weights[coo][2][0]);
				uint6 tmp7 = compute_engine_16(bottom[row+1][col  ], weights[coo][2][1]);
				uint6 tmp8 = compute_engine_16(bottom[row+1][col+1], weights[coo][2][2]);

				uint8 sum0 = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);
				FIX_FM_acc norm = batch_norm(sum0, bn_weight[coo], bn_bias[coo]);
				top[coo][row0][col0] = d + norm;
			}
		}
	}
}

void biconv32(uint32 bottom[32][32],
              uint32 weights[32][3][3],
              FIX_FM_acc top[32][32][32]
)
{

#pragma HLS array_partition variable=top dim=1 complete
#pragma HLS array_partition variable=weights dim=1 complete

        biconv_row:for(int row = 1; row < 31; row ++){
            biconv_col:for(int col = 1; col < 31; col ++) {
#pragma HLS PIPELINE
                biconv_coo:for (int coo = 0; coo < 32; coo ++) {
#pragma HLS UNROLL
                 uint6 tmp0 = compute_engine_32(bottom[row-1][col-1], weights[coo][0][0]);
                 uint6 tmp1 = compute_engine_32(bottom[row-1][col  ], weights[coo][0][1]);
                 uint6 tmp2 = compute_engine_32(bottom[row-1][col+1], weights[coo][0][2]);
                 uint6 tmp3 = compute_engine_32(bottom[row  ][col-1], weights[coo][1][0]);
                 uint6 tmp4 = compute_engine_32(bottom[row  ][col  ], weights[coo][1][1]);
                 uint6 tmp5 = compute_engine_32(bottom[row  ][col+1], weights[coo][1][2]);
                 uint6 tmp6 = compute_engine_32(bottom[row+1][col-1], weights[coo][2][0]);
                 uint6 tmp7 = compute_engine_32(bottom[row+1][col  ], weights[coo][2][1]);
                 uint6 tmp8 = compute_engine_32(bottom[row+1][col+1], weights[coo][2][2]);
                top[coo][row][col] = sum_engine(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8);;
            }
        }
    }
}

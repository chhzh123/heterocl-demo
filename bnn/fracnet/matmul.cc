#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>


void matmul(FIX_FM_acc bottom[64],
				const FIX_WT weights[10][64],
				const FIX_WT bias[10],
				float top[10]
)
{
//#pragma HLS ARRAY_PARTITION variable=top complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1

      // Iterate over the columns of the B matrix
	FIX_FM_acc buf[10];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
	for(int coo = 0; coo < 10; coo ++) {
#pragma HLS UNROLL
		buf[coo] = bias[coo];
	}
	for(int cii = 0; cii < 64; cii++) {
#pragma HLS PIPELINE II=2
		for(int coo = 0; coo < 10; coo ++) {
#pragma HLS UNROLL
			buf[coo] = buf[coo] + bottom[cii] * weights[coo][cii];
		}
	}
	for(int coo = 0; coo < 10; coo ++) {
#pragma HLS PIPELINE
		top[coo] = buf[coo];
	}
}

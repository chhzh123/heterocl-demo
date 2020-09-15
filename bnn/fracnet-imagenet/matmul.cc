#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>


void matmul(FIX_FM_acc bottom[64],
                FIX_WT weights[10][64],
                FIX_WT bias[10],
                float top[10]
)
{
//#pragma HLS ARRAY_RESHAPE variable=bottom complete dim=1
#pragma HLS ARRAY_PARTITION variable=top complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
//#pragma HLS INTERFACE ap_fifo port=bottom
//#pragma HLS INTERFACE ap_fifo port=weights
//#pragma HLS INTERFACE ap_fifo port=bias
//#pragma HLS INTERFACE ap_fifo port=top

      // Iterate over the columns of the B matrix
    FIX_FM_acc buf[10];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1


//    	#pragma HLS PIPELINE
    	for(int cii = 0; cii < 64; cii++) {

#pragma HLS PIPELINE
        	FIX_FM_acc bt = bottom[cii];
    		for(int coo = 0; coo < 10; coo ++) {
#pragma HLS UNROLL
        	FIX_FM_acc d = buf[coo];
        	FIX_WT wt = weights[coo][cii];
        	FIX_FM_acc bw = bt * wt;
        	FIX_FM_acc buf_tmp = d + bw;
            buf[coo] = buf_tmp;
        }
    }
    for(int coo = 0; coo < 10; coo ++) {
#pragma HLS UNROLL
        top[coo] = buf[coo] + bias[coo];
    }
}

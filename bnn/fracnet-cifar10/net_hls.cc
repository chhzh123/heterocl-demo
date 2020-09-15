
#include "net_hls.h"
#include "pgconv64.h"
#include "weights.h"
#include <math.h>
#include <fstream>
#include <hls_math.h>
#include <ap_fixed.h>
#include <string.h>

using namespace std;


// feature map buffers

FIX_FM_acc fm_buf[4][4][64][10][10];
//FIX_FM_acc fm_buf_s[16][10][10];

uint64 input_buf[2][10][10]; // msb: buf[1] lsb: buf[0]
uint64 weights_buf0[16][3][3];

FIX_FM_acc out_buf0[16][10][10];
FIX_FM_acc out_buf1[16][10][10];

FIX_WT bn_weight_buf[16];
FIX_WT bn_bias_buf[16];

void fill_fm_buf(int row, int col) {
    for (int frow = 1; frow < 9; frow ++) {
        for (int fcol = 1; fcol < 9; fcol ++) {
        	uint64 t1, t0;
            for (int c = 0; c < 64; c ++) {
#pragma HLS UNROLL
                uint2 t = fm_buf[row][col][c][frow][fcol];
                t1[c] = t[1];
                t0[c] = t[0];
            }
            input_buf[1][frow][fcol] = t1;
            input_buf[0][frow][fcol] = t0;
        }
    }
}

template <unsigned CHANNEL>
void fill_fm_buf_bn(int row, int col, int c, int c_cat, const FIX_WT bn_weight[CHANNEL], const FIX_WT bn_bias[CHANNEL]) {
#pragma HLS ARRAY_PARTITION variable=bn_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=bn_bias complete dim=1
	for (int brow = 1; brow < 9; brow ++) {
		for (int bcol = 1; bcol < 9; bcol ++) {
#pragma HLS PIPELINE
			for (int b = 0; b < 16; b ++) {
#pragma HLS UNROLL
				FIX_FM_acc wt = bn_weight[c*16 + b];
				FIX_FM_acc bs = bn_bias[c*16 + b];
				FIX_FM_acc t = out_buf0[b][brow][bcol] + fm_buf[row][col][c_cat*16 + b][brow][bcol];
				FIX_FM_acc t0 = t*wt + bs;
				fm_buf[row][col][c_cat*16 + b][brow][bcol] = t0;
			}
		}
	}
}

void ResNet(  uint16 image_thermo[6*32*32],
                float result[10]
)
{
#pragma HLS ARRAY_PARTITION variable=fm_buf complete dim=3

#pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_buf0 complete dim=1

#pragma HLS ARRAY_PARTITION variable=out_buf1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf0 complete dim=1

//#pragma HLS ARRAY_PARTITION variable=image_thermo complete dim=1
#pragma HLS INTERFACE m_axi depth=131072 port=image_thermo offset=slave bundle=IMG

#pragma HLS INTERFACE m_axi depth=120 port=result offset=slave bundle=BUS32


#pragma HLS INTERFACE s_axilite register port=return

// image_thermo comes as uint16 in 6 groups to make 96 channels
// the weights is [96][3][3], 96 divides into 6 groups of 16 as well

	uint16 image_buf[6][32][32];
#pragma HLS ARRAY_PARTITION variable=image_buf complete dim=1

	for (int row = 0; row < 32; row ++) {
		for (int col = 0; col < 32; col ++) {
			for (int cbb = 0; cbb < 6; cbb ++) {
#pragma HLS PIPELINE
				image_buf[cbb][row][col] = image_thermo[cbb*1024+row*32+col];
			}
		}
	}

	int N_CII, N_CIO, N_COI, N_COO;

	////////////////// LAYER 0 ////////////////////////

	FIX_FM_acc conv1_out[16][32][32];
	uint16 conv1_weight_buf[16][3][3];

	N_CII = 96 / 16;
	N_COO = 16 / 16;

	for (int cii = 0; cii < N_CII; cii ++) {
		for (int row = 0; row < 3; row ++) {
			for (int col = 0; col < 3; col ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < 16; c ++) {
#pragma HLS UNROLL
					conv1_weight_buf[c][row][col] = conv1_weight[cii*16+c][row][col];
				}
			}
		}
		biconv16(image_buf[cii],
				conv1_weight_buf,
				conv1_out);
	}
	for (int row_b = 0; row_b < 4; row_b ++) {
		for (int col_b = 0; col_b < 4; col_b ++) {
			for (int brow = 1; brow < 9; brow ++) {
				for (int bcol = 1; bcol < 9; bcol ++) {
#pragma HLS PIPELINE
					for (int c = 0; c < 16; c ++) {
#pragma HLS UNROLL
						fm_buf[row_b][col_b][c][brow][bcol] = conv1_out[c][row_b*8+brow][col_b*8+bcol]*bn1_weight[c] + bn1_bias[c];
					}
				}
			}
		}
	}



	////////////////// LAYER 1-0 ////////////////////////

	N_CII = 16 / 16;
	N_CIO = 16 / 16;
	N_COI = 16 / 16;
	N_COO = 16 / 16;

	for (int row = 0; row < 4; row ++) {
		for (int col = 0; col < 4; col ++) {
			fill_fm_buf(row, col);
			pgconv64<16>(input_buf[1],
						input_buf[0],
						0,
						layer1_0_conv1_weight,
						layer1_0_conv1_threshold,
						layer1_0_bn1_weight,
						layer1_0_bn1_bias,
						layer1_0_rprelu1_shift_x_bias,
						layer1_0_rprelu1_shift_y_bias,
						layer1_0_rprelu1_prelu_weight,
						out_buf0);
			fill_fm_buf_bn<16>(row, col, 0, 0,
						layer1_0_bn3_weight,
						layer1_0_bn3_bias);
			fill_fm_buf(row, col);
			pgconv64<16>(input_buf[1],
						input_buf[0],
						0,
						layer1_0_conv2_weight,
						layer1_0_conv2_threshold,
						layer1_0_bn2_weight,
						layer1_0_bn2_bias,
						layer1_0_rprelu2_shift_x_bias,
						layer1_0_rprelu2_shift_y_bias,
						layer1_0_rprelu2_prelu_weight,
						out_buf0);
			fill_fm_buf_bn<16>(row, col, 0, 0,
						layer1_0_bn4_weight,
						layer1_0_bn4_bias);
		}
	}

	////////////////// LAYER 1-1 ////////////////////////

	N_CII = 16 / 16;
	N_CIO = 16 / 16;
	N_COI = 16 / 16;
	N_COO = 16 / 16;

	for (int row = 0; row < 4; row ++) {
		for (int col = 0; col < 4; col ++) {
			fill_fm_buf(row, col);
			pgconv64<16>(input_buf[1],
						input_buf[0],
						0,
						layer1_1_conv1_weight,
						layer1_1_conv1_threshold,
						layer1_1_bn1_weight,
						layer1_1_bn1_bias,
						layer1_1_rprelu1_shift_x_bias,
						layer1_1_rprelu1_shift_y_bias,
						layer1_1_rprelu1_prelu_weight,
						out_buf0);
			fill_fm_buf_bn<16>(row, col, 0, 0,
						layer1_1_bn3_weight,
						layer1_1_bn3_bias);
			fill_fm_buf(row, col);
			pgconv64<16>(input_buf[1],
						input_buf[0],
						0,
						layer1_1_conv2_weight,
						layer1_1_conv2_threshold,
						layer1_1_bn2_weight,
						layer1_1_bn2_bias,
						layer1_1_rprelu2_shift_x_bias,
						layer1_1_rprelu2_shift_y_bias,
						layer1_1_rprelu2_prelu_weight,
						out_buf0);
			fill_fm_buf_bn<16>(row, col, 0, 0,
						layer1_1_bn4_weight,
						layer1_1_bn4_bias);
		}
	}

   ////////////////// LAYER 1-2 ////////////////////////

	N_CII = 16 / 16;
	N_CIO = 16 / 16;
	N_COI = 16 / 16;
	N_COO = 16 / 16;

	for (int row = 0; row < 4; row ++) {
		for (int col = 0; col < 4; col ++) {
			fill_fm_buf(row, col);
			pgconv64<16>(input_buf[1],
						input_buf[0],
						0,
						layer1_2_conv1_weight,
						layer1_2_conv1_threshold,
						layer1_2_bn1_weight,
						layer1_2_bn1_bias,
						layer1_2_rprelu1_shift_x_bias,
						layer1_2_rprelu1_shift_y_bias,
						layer1_2_rprelu1_prelu_weight,
						out_buf0);
			fill_fm_buf_bn<16>(row, col, 0, 0,
						layer1_2_bn3_weight,
						layer1_2_bn3_bias);
			fill_fm_buf(row, col);
			pgconv64<16>(input_buf[1],
						input_buf[0],
						0,
						layer1_2_conv2_weight,
						layer1_2_conv2_threshold,
						layer1_2_bn2_weight,
						layer1_2_bn2_bias,
						layer1_2_rprelu2_shift_x_bias,
						layer1_2_rprelu2_shift_y_bias,
						layer1_2_rprelu2_prelu_weight,
						out_buf0);
			fill_fm_buf_bn<16>(row, col, 0, 0,
						layer1_2_bn4_weight,
						layer1_2_bn4_bias);
		}
	}

	////////////////// LAYER 2-0 ////////////////////////

	N_CII = 16 / 16;
	N_CIO = 32 / 16;
	N_COI = 32 / 16;
	N_COO = 32 / 16;

	for (int cio = 0; cio < N_CIO; cio ++) {
		for (int row = 0; row < 2; row ++) {
			for (int col = 0; col < 2; col ++) {
				for (int cii = 0; cii < N_CII; cii ++) {
					for (int row0 = 0; row0 < 2; row0 ++) {
						for (int col0 = 0; col0 < 2; col0 ++) {
							fill_fm_buf(row*2 + row0, col*2 + col0);
							pgconv64s2<16>(input_buf[1],
									input_buf[0],
									cii,
									row0,
									col0,
									layer2_0_conv1_weight,
									layer2_0_conv1_threshold,
									layer2_0_bn1_weight,
									layer2_0_bn1_bias,
									layer2_0_rprelu1_shift_x_bias,
									layer2_0_rprelu1_shift_y_bias,
									layer2_0_rprelu1_prelu_weight,
									out_buf0);
						}
					}
				}
				fill_fm_buf_bn<32>(row, col, cio, 0,
							layer2_0_bn3_weight,
							layer2_0_bn3_bias);
			}
		}
	}
	for (int row = 0; row < 2; row ++) {
		for (int col = 0; col < 2; col ++) {
			fill_fm_buf(row, col);
			for (int coo = 0; coo < N_COO; coo ++) {
				for (int coi = 0; coi < N_COI; coi ++) {
					pgconv64<32>(input_buf[1],
							input_buf[0],
							coi,
							layer2_0_conv2_weight,
							layer2_0_conv2_threshold,
							layer2_0_bn2_weight,
							layer2_0_bn2_bias,
							layer2_0_rprelu2_shift_x_bias,
							layer2_0_rprelu2_shift_y_bias,
							layer2_0_rprelu2_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<32>(row, col, coo, coo,
							layer2_0_bn4_weight,
							layer2_0_bn4_bias);
			}
		}
	}

	////////////////// LAYER 2-1 ////////////////////////

	N_CII = 32 / 16;
	N_CIO = 32 / 16;
	N_COI = 32 / 16;
	N_COO = 32 / 16;


	for (int row = 0; row < 2; row ++) {
		for (int col = 0; col < 2; col ++) {
			fill_fm_buf(row, col);
			for (int cio = 0; cio < N_CIO; cio ++) {
				for (int cii = 0; cii < N_CII; cii ++) {
					pgconv64<32>(input_buf[1],
							input_buf[0],
							cii,
							layer2_1_conv1_weight,
							layer2_1_conv1_threshold,
							layer2_1_bn1_weight,
							layer2_1_bn1_bias,
							layer2_1_rprelu1_shift_x_bias,
							layer2_1_rprelu1_shift_y_bias,
							layer2_1_rprelu1_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<32>(row, col, cio, cio,
							layer2_1_bn3_weight,
							layer2_1_bn3_bias);
			}
			fill_fm_buf(row, col);
			for (int coo = 0; coo < N_COO; coo ++) {
				for (int coi = 0; coi < N_COI; coi ++) {
					pgconv64<32>(input_buf[1],
							input_buf[0],
							coi,
							layer2_1_conv2_weight,
							layer2_1_conv2_threshold,
							layer2_1_bn2_weight,
							layer2_1_bn2_bias,
							layer2_1_rprelu2_shift_x_bias,
							layer2_1_rprelu2_shift_y_bias,
							layer2_1_rprelu2_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<32>(row, col, coo, coo,
							layer2_1_bn4_weight,
							layer2_1_bn4_bias);
			}
		}
	}

   ////////////////// LAYER 2-2 ////////////////////////

	N_CII = 32 / 16;
	N_CIO = 32 / 16;
	N_COI = 32 / 16;
	N_COO = 32 / 16;

	for (int row = 0; row < 2; row ++) {
		for (int col = 0; col < 2; col ++) {
			fill_fm_buf(row, col);
			for (int cio = 0; cio < N_CIO; cio ++) {
				for (int cii = 0; cii < N_CII; cii ++) {
					pgconv64<32>(input_buf[1],
							input_buf[0],
							cii,
							layer2_2_conv1_weight,
							layer2_2_conv1_threshold,
							layer2_2_bn1_weight,
							layer2_2_bn1_bias,
							layer2_2_rprelu1_shift_x_bias,
							layer2_2_rprelu1_shift_y_bias,
							layer2_2_rprelu1_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<32>(row, col, cio, cio,
							layer2_2_bn3_weight,
							layer2_2_bn3_bias);
			}
			fill_fm_buf(row, col);
			for (int coo = 0; coo < N_COO; coo ++) {
				for (int coi = 0; coi < N_COI; coi ++) {
					pgconv64<32>(input_buf[1],
							input_buf[0],
							coi,
							layer2_2_conv2_weight,
							layer2_2_conv2_threshold,
							layer2_2_bn2_weight,
							layer2_2_bn2_bias,
							layer2_2_rprelu2_shift_x_bias,
							layer2_2_rprelu2_shift_y_bias,
							layer2_2_rprelu2_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<32>(row, col, coo, coo,
							layer2_2_bn4_weight,
							layer2_2_bn4_bias);
			}
		}
	}

	////////////////// LAYER 3-0 ////////////////////////

	N_CII = 32 / 16;
	N_CIO = 64 / 16;
	N_COI = 64 / 16;
	N_COO = 64 / 16;

	for (int cio = 0; cio < N_CIO; cio ++) {
		for (int row = 0; row < 1; row ++) {
			for (int col = 0; col < 1; col ++) {
				for (int cii = 0; cii < N_CII; cii ++) {
					for (int row0 = 0; row0 < 2; row0 ++) {
						for (int col0 = 0; col0 < 2; col0 ++) {
							fill_fm_buf(row*2 + row0, col*2 + col0);
							pgconv64s2<32>(input_buf[1],
									input_buf[0],
									cii,
									row0,
									col0,
									layer3_0_conv1_weight,
									layer3_0_conv1_threshold,
									layer3_0_bn1_weight,
									layer3_0_bn1_bias,
									layer3_0_rprelu1_shift_x_bias,
									layer3_0_rprelu1_shift_y_bias,
									layer3_0_rprelu1_prelu_weight,
									out_buf0);
						}
					}
				}
				int cc = cio;
				if (cio > 2) {
					cc = cio - 2;
				}
				fill_fm_buf_bn<64>(row, col, cio, 0,
							layer3_0_bn3_weight,
							layer3_0_bn3_bias);
			}
		}
	}

	for (int row = 0; row < 1; row ++) {
		for (int col = 0; col < 1; col ++) {
			fill_fm_buf(row, col);
			for (int coo = 0; coo < N_COO; coo ++) {
				for (int coi = 0; coi < N_COI; coi ++) {
					pgconv64<64>(input_buf[1],
							input_buf[0],
							coi,
							layer3_0_conv2_weight,
							layer3_0_conv2_threshold,
							layer3_0_bn2_weight,
							layer3_0_bn2_bias,
							layer3_0_rprelu2_shift_x_bias,
							layer3_0_rprelu2_shift_y_bias,
							layer3_0_rprelu2_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<64>(row, col, coo, coo,
							layer3_0_bn4_weight,
							layer3_0_bn4_bias);
			}
		}
	}

	////////////////// LAYER 3-1 ////////////////////////

	N_CII = 64 / 16;
	N_CIO = 64 / 16;
	N_COI = 64 / 16;
	N_COO = 64 / 16;

	for (int row = 0; row < 1; row ++) {
		for (int col = 0; col < 1; col ++) {
			fill_fm_buf(row, col);
			for (int cio = 0; cio < N_CIO; cio ++) {
				for (int cii = 0; cii < N_CII; cii ++) {
					pgconv64<64>(input_buf[1],
							input_buf[0],
							cii,
							layer3_1_conv1_weight,
							layer3_1_conv1_threshold,
							layer3_1_bn1_weight,
							layer3_1_bn1_bias,
							layer3_1_rprelu1_shift_x_bias,
							layer3_1_rprelu1_shift_y_bias,
							layer3_1_rprelu1_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<64>(row, col, cio, cio,
							layer3_1_bn3_weight,
							layer3_1_bn3_bias);
			}
			fill_fm_buf(row, col);
			for (int coo = 0; coo < N_COO; coo ++) {
				for (int coi = 0; coi < N_COI; coi ++) {
					pgconv64<64>(input_buf[1],
							input_buf[0],
							coi,
							layer3_1_conv2_weight,
							layer3_1_conv2_threshold,
							layer3_1_bn2_weight,
							layer3_1_bn2_bias,
							layer3_1_rprelu2_shift_x_bias,
							layer3_1_rprelu2_shift_y_bias,
							layer3_1_rprelu2_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<64>(row, col, coo, coo,
							layer3_1_bn4_weight,
							layer3_1_bn4_bias);
			}
		}
	}

   ////////////////// LAYER 3-2 ////////////////////////

	N_CII = 64 / 16;
	N_CIO = 64 / 16;
	N_COI = 64 / 16;
	N_COO = 64 / 16;

	for (int row = 0; row < 1; row ++) {
		for (int col = 0; col < 1; col ++) {
			fill_fm_buf(row, col);
			for (int cio = 0; cio < N_CIO; cio ++) {
				for (int cii = 0; cii < N_CII; cii ++) {
					pgconv64<64>(input_buf[1],
							input_buf[0],
							cii,
							layer3_2_conv1_weight,
							layer3_2_conv1_threshold,
							layer3_2_bn1_weight,
							layer3_2_bn1_bias,
							layer3_2_rprelu1_shift_x_bias,
							layer3_2_rprelu1_shift_y_bias,
							layer3_2_rprelu1_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<64>(row, col, cio, cio,
							layer3_2_bn3_weight,
							layer3_2_bn3_bias);
			}
			fill_fm_buf(row, col);
			for (int coo = 0; coo < N_COO; coo ++) {
				for (int coi = 0; coi < N_COI; coi ++) {
					pgconv64<64>(input_buf[1],
							input_buf[0],
							coi,
							layer3_2_conv2_weight,
							layer3_2_conv2_threshold,
							layer3_2_bn2_weight,
							layer3_2_bn2_bias,
							layer3_2_rprelu2_shift_x_bias,
							layer3_2_rprelu2_shift_y_bias,
							layer3_2_rprelu2_prelu_weight,
							out_buf0);
				}
				fill_fm_buf_bn<64>(row, col, coo, coo,
							layer3_2_bn4_weight,
							layer3_2_bn4_bias);
			}
		}
	}

	FIX_FM_acc linear_buf[64];
#pragma HLS ARRAY_PARTITION variable=linear_buf complete dim=1
	for (int i = 1; i < 9; i ++) {
		for (int j = 1; j < 9; j ++) {
#pragma HLS PIPELINE
			for (int c = 0; c < 64; c ++) {
#pragma HLS UNROLL	
				FIX_FM_acc d = linear_buf[c];
				linear_buf[c] = d + fm_buf[0][0][c][i][j];
			}
		}
	}

	for (int c = 0; c < 64; c ++) {
#pragma HLS UNROLL
		FIX_FM_acc d = linear_buf[c];
		linear_buf[c] = d/64;
	}

	matmul(linear_buf, linear_weight, linear_bias, result);

	return;
}

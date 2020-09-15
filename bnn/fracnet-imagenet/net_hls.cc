
#include "net_hls.h"
#include <math.h>
#include <fstream>
#include <hls_math.h>
#include <ap_fixed.h>
#include <string.h>

using namespace std;
#define DDR_OFFSET 0
#define CONV_WT_DEPTH 64
#define PG_DEPTH 64
#define BLK_DEPTH 32

#define DDR_OFFSET 0
#define WEIGHT_DEPTH 64
#define BLK_DEPTH 32

// feature map buffers
FIX_FM FM_buf0[32][9][9];
FIX_FM FM_buf1[32][9][9];
uint64 pg_buf_all[12996]; //114*114
uint64 pg_buf0[9][9];
FIX_FM_acc FM_buf_acc0[BLK_DEPTH][9][9];
// FIX_FM_acc FM_buf_acc1[BLK_DEPTH][11][11];
// FIX_FM_acc FM_buf_acc2[BLK_DEPTH][11][11];

// weight buffers
uint64 weight_buf_1x1[4][BLK_DEPTH];
uint64 weight_buf_3x3[4][BLK_DEPTH][3][3];
FIX_WT thres_buf[4][BLK_DEPTH];
FIX_WT bn_weight_buf[4][BLK_DEPTH];
FIX_WT bn_bias_buf[4][BLK_DEPTH];
FIX_WT relu_shiftx_buf[2][BLK_DEPTH];
FIX_WT relu_shifty_buf[2][BLK_DEPTH];
FIX_WT relu_weight_buf[2][BLK_DEPTH];

void copy_input_layer_buf_to_DDR(uint512 *dest, int row_offset, int col_offset)
{
#pragma HLS ARRAY_PARTITION variable = FM_buf_acc complete dim = 1

	uint512 *dest_ptr = dest + (row_offset * 4 + 1) * 114 + col_offset * 4 + 1;

	for (int row = 0; row < 4; row++)
	{
		for (int col = 0; col < 4; col++)
		{
#pragma HLS pipeline
			uint512 DATA = 0;
			for (int c = 0; c < 32; c++)
			{
#pragma HLS unroll
				FIX_FM_acc d = FM_buf_acc0[c][row][col];
				DATA.range(FM_ACC_RG + c * 16, c * 16) = d.range(FM_ACC_RG, 0);
			}
			dest_ptr[col].range(511, 0) = DATA.range(511, 0);
		}
		dest_ptr += 114;
	}
}

void load_buf_from_buf_all(int row, int col, int coff_row, int coff_col, int map_dim)
{
#pragma HLS ARRAY_PARTITION variable = pg_buf complete dim = 1
	// The second dimension is 114*114
	// coff_row and coff_col are for channel multipliers
	// for example, 256*28*28, first 28*28 is the first 64 channels
	// therefore the row offset is (coff_row*map_dim + row*7)*114 + (coff_col*map_dim + col*7)
	// it goes
	int index = (coff_row * map_dim + row * 7) * 114 + (coff_col * map_dim + col * 7);
	for (int h = 0; h < 9; h++)
	{
		for (int w = 0; w < 9; w++)
		{
#pragma HLS pipeline
			for (int c = 0; c < 64; c++)
			{
#pragma HLS unroll
				uint1 tmp = pg_buf_all[c][index + w];
				pg_buf0[h][w][c] = tmp;
			}
		}
		index += 114;
	}
}

void load_buf_from_DDR(uint512 *src, int src_offset, FIX_FM dest[32][9][9], int row_offset, int col_offset, int ch_offset)
{
#pragma HLS ARRAY_PARTITION variable = dest complete dim = 1
	// ddr[2][16][16][32][7][7]
	// ddr[ch_off][row_off][col_off][c][h][w]
	uint512 *src_ptr = src + src_offset * 415872 + ch_offset * 114 * 114 + row_offset * 7 * 114 + col_offset * 7;

	for (int h = 1; h < 8; h++)
	{
		for (int w = 1; w < 8; w++)
		{
#pragma HLS pipeline
			uint512 DATA = src_ptr[w - 1];
			for (int c = 0; c < 32; c++)
			{
#pragma HLS unroll
				FIX_FM d = 0;
				d.range(FM_RG, 0) = DATA.range(FM_RG + c * 16, c * 16);
				dest[c][h][w] = d;
			}
		}
		src_ptr += 114;
	}
}

void copy_buf_to_DDR(uint512 *dest, int dest_offset, FIX_FM src[32][9][9], int row_offset, int col_offset, int ch_offset)
{
#pragma HLS ARRAY_PARTITION variable = dest complete dim = 1
	// ddr[2][16][16][32][7][7]
	// ddr[ch_off][row_off][col_off][c][h][w]
	uint512 *dest_ptr = dest + dest_offset * 415872 + ch_offset * 114 * 114 + row_offset * 7 * 114 + col_offset * 7;

	for (int h = 1; h < 8; h++)
	{
		for (int w = 1; w < 8; w++)
		{
#pragma HLS pipeline
			uint512 DATA = 0;
			for (int c = 0; c < 32; c++)
			{
#pragma HLS unroll
				DATA.range(FM_RG + c * 16, c * 16) = src[c][h][w];
			}
			dest_ptr[w - 1].range(511, 0) = DATA.range(511, 0);
		}
		dest_ptr += 114;
	}
}

void load_weight_1x1_from_axi(uint64 dest[32], uint512 src, int coff)
{
	uint512 DATA = 0;
	DATA.range(511, 0) = src.range(511, 0);
	for (int c = 0; c < 8; c++)
	{
#pragma HLS unroll
		dest[coff * 8 + c] = DATA.range(63 + c * 64, c * 64);
	}
}

void load_weight_3x3_from_axi(uint64 dest[32][3][3], uint512 src[1000][3][3], int index)
{
#pragma HLS ARRAY_PARTITION variable = dest complete dim = 1

	// index: should be which the channel offset of each layer in groups of 4
	uint512 src_buf[4][3][3];
	for (int cc = 0; cc < 4; cc++)
	{
		for (int m = 0; m < 3; m++)
		{
			for (int n = 0; n < 3; n++)
			{
#pragma HLS pipeline
				uint512 DATA = 0;
				DATA.range(511, 0) = src[index + cc][m][n].range(511, 0);
				src_buf[cc][m][n] = DATA;
			}
		}
	}
	for (int m = 0; m < 3; m++)
	{
		for (int n = 0; n < 3; n++)
		{
			for (int cc = 0; cc < 4; cc++)
			{
#pragma HLS pipeline
				uint512 DATA = src_buf[cc][m][n];
				for (int c = 0; c < 8; c++)
				{
#pragma HLS unroll
					dest[cc * 8 + c][m][n] = DATA.range(63 + c * 64, c * 64);
				}
			}
		}
	}
}

void load_1x1_from_axi(FIX_WT dest[32], uint512 src)
{
	uint512 DATA = 0;
	DATA.range(511, 0) = src.range(511, 0);
	for (int c = 0; c < 32; c++)
	{
#pragma HLS unroll
		dest[c] = DATA.range(WT_RG + c * 16, c * 16);
	}
}

void clear_buffer(FIX_FM_acc buf[32][32][32])
{
	for (int h = 0; h < 32; h += 2)
	{
		for (int w = 0; w < 32; w++)
		{
#pragma HLS pipeline
			for (int c = 0; c < 32; c++)
			{
#pragma HLS unroll
				buf[c][h][w] = 0;
				buf[c][h + 1][w] = 0;
			}
		}
	}
}

inline FIX_FM_acc sum_engine(FIX_FM t0,
							 FIX_FM t1,
							 FIX_FM t2,
							 FIX_FM t3,
							 FIX_FM t4,
							 FIX_FM t5,
							 FIX_FM t6)
{
#pragma HLS PIPELINE
	FIX_FM_acc add0, add1, add2, add3, add4, add5, add6;

	add0 = t0 + t1;
	add1 = t2 + t3;
	add2 = t4 + t5;

	add3 = add0 + add1;
	add4 = add2 + t6;

	return add3 + add4;
}

FIX_FM_acc avgpool_7x7(FIX_FM buf[9][9])
{
	FIX_FM_acc sum = 0;
	for (int row = 1; row < 8; row++)
	{
#pragma HLS pipeline II = 4
		sum += sum_engine(buf[row][1], buf[row][2], buf[row][3], buf[row][4], buf[row][5], buf[row][6], buf[row][7]);
	}
	return sum / 32; // should divide by 49
}

void load_others(uint512 weights_all[10000],
				 int weights_all_index)
{
	uint512 DATA[8];
#pragma HLS ARRAY_PARTITION variable = DATA complete dim = 1
	for (int i = 0; i < 8; i++)
	{
#pragma HLS pipeline
		DATA[i].range(511, 0) = weights_all[weights_all_index + i].range(511, 0);
	}
	for (int c = 0; c < 32; c++)
	{
#pragma HLS unroll
		bn_weight_buf[0][c] = DATA[0].range(WT_RG + c * 16, c * 16);
		bn_bias_buf[0][c] = DATA[1].range(WT_RG + c * 16, c * 16);
		thres_buf[0][c] = DATA[2].range(WT_RG + c * 16, c * 16);
		relu_shiftx_buf[0][c] = DATA[3].range(WT_RG + c * 16, c * 16);
		relu_shifty_buf[0][c] = DATA[4].range(WT_RG + c * 16, c * 16);
		relu_weight_buf[0][c] = DATA[5].range(WT_RG + c * 16, c * 16);
		bn_weight_buf[0][c] = DATA[6].range(WT_RG + c * 16, c * 16);
		bn_bias_buf[0][c] = DATA[7].range(WT_RG + c * 16, c * 16);
	}
}

void load_weights_3x3_all(uint512 conv_weight_3x3_all[1000][3][3],
						  int weight_3x3_index,
						  uint512 weights_all[10000],
						  int weights_all_index)
{
	load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all, weight_3x3_index);
	//    load_1x1_from_axi(bn_weight_buf[0], weights_all[weights_all_index]);
	//    load_1x1_from_axi(bn_bias_buf[0], weights_all[weights_all_index + 1]);
	//    load_1x1_from_axi(thres_buf[0], weights_all[weights_all_index + 2]);
	//    load_1x1_from_axi(relu_shiftx_buf[0], weights_all[weights_all_index + 3]);
	//    load_1x1_from_axi(relu_shifty_buf[0], weights_all[weights_all_index + 4]);
	//    load_1x1_from_axi(relu_weight_buf[0], weights_all[weights_all_index + 5]);
	//    load_1x1_from_axi(bn_weight_buf[1], weights_all[weights_all_index + 6]);
	//    load_1x1_from_axi(bn_bias_buf[1], weights_all[weights_all_index + 7]);
	load_others(weights_all, weights_all_index);
}

void load_weights_1x1_all(uint512 conv_weight_1x1_all[1000],
						  int weight_1x1_index, uint512 weights_all[10000],
						  int weights_all_index)
{
	for (int coff = 0; coff < 4; coff++)
	{
		load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + coff], coff);
	}
	//    load_1x1_from_axi(bn_weight_buf[0], weights_all[weights_all_index]);
	//    load_1x1_from_axi(bn_bias_buf[0], weights_all[weights_all_index + 1]);
	//    load_1x1_from_axi(thres_buf[0], weights_all[weights_all_index + 2]);
	//    load_1x1_from_axi(relu_shiftx_buf[0], weights_all[weights_all_index + 3]);
	//    load_1x1_from_axi(relu_shifty_buf[0], weights_all[weights_all_index + 4]);
	//    load_1x1_from_axi(relu_weight_buf[0], weights_all[weights_all_index + 5]);
	//    load_1x1_from_axi(bn_weight_buf[1], weights_all[weights_all_index + 6]);
	//    load_1x1_from_axi(bn_bias_buf[1], weights_all[weights_all_index + 7]);
	load_others(weights_all, weights_all_index);
}

inline FIX_FM_acc batch_norm(uint8 sum, FIX_WT weight, FIX_WT bias)
{
	return sum * weight + bias;
}

inline FIX_FM_acc relu(FIX_FM_acc norm, FIX_WT shiftx, FIX_WT shifty, FIX_WT weight)
{
	if (norm > 0)
	{
		return norm + shifty;
	}
	else
	{
		return norm * weight + shifty;
	}
}

void store_bufs_organize(uint512 *ddr_ptr, int dest_offset, int row_offset, int col_offset, int ch_offset, int coff_row, int coff_col, int map_dim, int stride)
{

#pragma HLS array_partition variable = bn_weights_buf dim = 2 complete
#pragma HLS array_partition variable = bn_bias_buf dim = 2 complete
#pragma HLS array_partition variable = relu_shiftx_buf dim = 2 complete
#pragma HLS array_partition variable = relu_shifty_buf dim = 2 complete
#pragma HLS array_partition variable = relu_weights_buf dim = 2 complete
	// if the stride is 2
	// input buffer is meaningful stuff at 1, 3, 5, 7
	int s;
	if (stride == 2)
	{
		s = 4;
	}
	else
	{
		s = 7;
	}
	uint512 *dest_ptr = ddr_ptr + dest_offset * 415872 + ch_offset * 114 * 114 + row_offset * 7 * 114 + col_offset * 7;
	int index = (coff_row * map_dim + row_offset * 7) * 114 + (coff_col * map_dim + col_offset * 7);
	for (int row0 = 0; row0 < s; row0++)
	{
		for (int col0 = 0; col0 < s; col0++)
		{
#pragma HLS pipeline
			int row, col;
			if (stride == 2)
			{
				row = row0 * 2 + 1;
				col = col0 * 2 + 1;
			}
			else
			{
				row = row0 + 1;
				col = col0 + 1;
			}
			uint512 DATA = 0;
			for (int c = 0; c < 32; c++)
			{
#pragma HLS unroll
				FIX_FM ds = FM_buf0[c][row][col];
				FIX_FM_acc fm = FM_buf_acc0[c][row][col];
				FIX_FM_acc d0 = batch_norm(fm, bn_weight_buf[0][c], bn_bias_buf[0][c]);
				FIX_FM_acc rl = relu(d0, relu_shiftx_buf[0], relu_shifty_buf[0], relu_weight_buf[0]);
				FIX_FM_acc d1 = rl + ds;
				FIX_FM_acc r = batch_norm(d1, bn_weight_buf[1][c], bn_bias_buf[1][c]);
				DATA.range(FM_RG + c * 16, c * 16) = r;
				uint1 sn;
				if (r > 0)
				{
					sn = 1;
				}
				else
				{
					sn = 0;
				}
				pg_buf_all[index + col][c + (ch_offset % 2) * 32] = sn;
			}
			dest_ptr[col - 1].range(511, 0) = DATA.range(511, 0);
		}
		dest_ptr += 114;
		index += 114;
	}
}

//void store_bufs_organize_s2(uint512* ddr_ptr, int dest_offset, int row_offset, int col_offset, int ch_offset, int coff_row, int coff_col, int map_dim)
//{
//    // if the stride is 2
//    // input buffer is meaningful stuff at 1, 3, 5, 7
//
//    uint512* dest_ptr = ddr_ptr + dest_offset*415872 + ch_offset*114*114 + row_offset*4*114 + col_offset*4;
//    int index = (coff_row*map_dim + row_offset*4)*114 + (coff_col*map_dim + col_offset*4);
//    for(int row0 = 0; row0 < 4; row0 ++) {
//        for(int col0 = 0; col0 < 4; col0 ++) {
//#pragma HLS pipeline
//          int row = row0*2 + 1;
//          int col = col0*2 + 1;
//          uint512 DATA = 0;
//            for(int c = 0; c < 32; c ++) {
//#pragma HLS unroll
//              FIX_FM ds = FM_buf0[c][row][col];
//                FIX_WT wt = bn_weight_buf[1][c];
//                FIX_WT bs = bn_bias_buf[1][c];
//                FIX_FM_acc d = FM_buf_acc0[c][row][col] + ds;
//                FIX_FM_acc dwt = d*wt;
//                FIX_FM_acc r = dwt + bs;
//                DATA.range(FM_RG + c*16, c*16) = r;
//                uint1 sn;
//                if (r > 0) {
//                    sn = 1;
//                } else {
//                    sn = 0;
//                }
//                pg_buf_all[c+(ch_offset%2)*32][index + col] = sn;
//            }
//            dest_ptr[col-1].range(511, 0) = DATA.range(511, 0);
//        }
//        dest_ptr += 114;
//        index += 114;
//    }
//}

void load_input(int row, int col, int c, uint64 buf[9][9], uint32 img[3 * 226 * 226])
{
	for (int mm = 0; mm < 9; mm++)
	{
		for (int nn = 0; nn < 9; nn++)
		{
#pragma HLS pipeline
			// image: 6*226*226
			// the stride 2 is tricky
			// populate the img buffer as follows:
			// 0:8, 8:16 and so on upto 8*27=126:224
			// the result with then have
			// (1, 3, 5, 7), (9, 11, 13, 15) and so on
			int img_index = c * 51076 + (col * 7 + mm) * 226 + (row * 7 + nn);
			buf[mm][nn].range(31, 0) = img[img_index].range(31, 0);
		}
	}
}

void FracNet(uint32 image_thermo[3 * 226 * 226],

			 uint512 conv_weight_1x1_all[1000],
			 uint512 conv_weight_3x3_all[1000][3][3],
			 uint512 weights_all[10000],
			 uint512 linear_weight_all[16000][2],
			 uint512 linear_bias_all[100],

			 uint512 *DDR_buff_merge,

			 float out[1000])
{
//#pragma HLS ARRAY_PARTITION variable=pg_buf_full complete dim=1
#pragma HLS ARRAY_PARTITION variable = FM_buf_acc0 complete dim = 1

#pragma HLS INTERFACE m_axi depth = 153228 port = image_thermo offset = slave bundle = IMG

#pragma HLS INTERFACE m_axi depth = 32000 port = conv_weight_1x1_all offset = slave bundle = BUS512
#pragma HLS INTERFACE m_axi depth = 32000 port = conv_weight_3x3_all offset = slave bundle = BUS512
#pragma HLS INTERFACE m_axi depth = 32000 port = weights_all offset = slave bundle = BUS512
#pragma HLS INTERFACE m_axi depth = 32000 port = linear_weight_all offset = slave bundle = BUS512
#pragma HLS INTERFACE m_axi depth = 32000 port = linear_bias_all offset = slave bundle = BUS512

#pragma HLS INTERFACE m_axi depth = 32000 port = out offset = slave bundle = BUS32

// ddr_buf[64][16][16][7][7]
#pragma HLS INTERFACE m_axi depth = 861184 port = DDR_buff_merge offset = slave bundle = DDR512

#pragma HLS INTERFACE s_axilite register port = return

//#pragma HLS ALLOCATION instances=biconv16                     limit=1 function
#pragma HLS ALLOCATION instances = pgconv64_1x1_1bit limit = 1 function
#pragma HLS ALLOCATION instances = pgconv64_1bit limit = 1 function
#pragma HLS ALLOCATION instances = matmul limit = 1 function
#pragma HLS ALLOCATION instances = store_bufs_organize limit = 1 function

	int N_CII, N_CIO, N_COI, N_COO, N_SPI, N_SPO, N_BLK, stride, off_row, off_col;
	int weight_3x3_index, weight_1x1_index, weights_all_index;

	N_BLK = BLK_DEPTH / (512 / WEIGHT_DEPTH);

	weight_3x3_index = 0;
	weight_1x1_index = 0;
	weights_all_index = 0;
	//    bn_bias_index = 0;
	//    bn_weight_index = 0;
	//    relu_shiftx_index = 0;
	//    relu_shifty_index = 0;
	//    relu_weight_index = 0;

	///////////////////////////// INPUT LAYER ////////////////////////////
	///////////////////////////// 32 112 112 ////////////////////////////

	uint64 conv1_weights[3][32][3][3];
	FIX_WT conv1_bn_weights[32];
	FIX_WT conv1_bn_bias[32];

	for (int b = 0; b < 2; b++)
	{
		for (int c = 0; c < 3; c++)
		{
			for (int m = 0; m < 3; m++)
			{
				for (int n = 0; n < 3; n++)
				{
#pragma HLS pipeline
					uint512 DATA = 0;
					DATA.range(511, 0) = conv_weight_3x3_all[b * 3 + c][m][n].range(511, 0);
					for (int cc = 0; cc < 32; cc++)
					{
#pragma HLS unroll
						conv1_weights[c][cc][m][n].range(15 + b * 16, b * 16) = DATA.range(15 + cc * 16, cc * 16);
					}
				}
			}
		}
	}
	weight_3x3_index += 6;
	//    for (int c = 0; c < 3; c ++) {
	//#pragma HLS pipeline
	//        uint512 DATA = 0;
	//        DATA.range(511, 0) = weights_all[c].range(511, 0);
	//        for (int cc = 0; cc < 32; cc++) {
	//#pragma HLS unroll
	//            conv1_bn_weights[c][cc] = DATA.range(15+cc*16, cc*16);
	//        }
	//    }
	//    weights_all_index += 6;
	//    for (int c = 0; c < 6; c ++) {
	//#pragma HLS pipeline
	//      uint512 DATA = 0;
	//      DATA.range(511, 0) = weights_all[c].range(511, 0);
	//      for (int cc = 0; cc < 32; cc++) {
	//#pragma HLS unroll
	//          conv1_bn_bias[c][cc] = DATA.range(15+cc*16, cc*16);
	//      }
	//  }
	//    weights_all_index += 6;

	uint64 conv1_img[3][9][9];
#pragma HLS ARRAY_PARTITION variable = FM_buf_acc0 complete dim = 1

	stride = 2;
	load_input(0, 0, 0, conv1_img[0], image_thermo);
input_biconv:
	for (int row = 0; row < 28; row++)
	{
		for (int col = 0; col < 28; col++)
		{
			for (int c = 0; c < 3; c++)
			{
				if (c < 2)
				{
					load_input(row, col, c + 1, conv1_img[c + 1], image_thermo);
				}
				pgconv64_1bit(conv1_img[c], conv1_weights[0], thres_buf[0], FM_buf_acc0, stride);
			}
			if (col != 27)
			{
				load_input(row, col + 1, 0, conv1_img[0], image_thermo);
			}
			if (col == 27 && row != 27)
			{
				load_input(row + 1, 0, 0, conv1_img[0], image_thermo);
			}

			// this thing is stride 2
			// careful with rearrangement of fm
			copy_input_layer_buf_to_DDR(DDR_buff_merge, row, col);
			int buf_index = (row * 4 + 1) * 114 + col * 4 + 1;
			for (int mm = 0; mm < 4; mm++)
			{
				for (int nn = 0; nn < 4; nn++)
				{
#pragma HLS pipeline
					for (int c = 0; c < 32; c++)
					{
#pragma HLS unroll
						pg_buf_all[buf_index + nn][c] = (uint1)FM_buf_acc0[c][mm][nn];
					}
				}
				buf_index += 114;
			}
		}
	}

	/////////////////////////////// 64 112 112 ////////////////////////////

	N_CII = 1;
	N_CIO = 32 / BLK_DEPTH;
	N_COI = 1;
	N_COO = 64 / BLK_DEPTH;
	N_SPI = 112 / 7;
	N_SPO = 112 / 7;
	stride = 1;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					// row, col, coff_row, coff_col, map_dim
					load_buf_from_buf_all(row, col, 0, 0, 112);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, 0, 0, 112, 1);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 112);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, 0, 0, 112, 1);
			}
		}
	}

	/////////////////////////////// 128 56 56 ////////////////////////////

	N_CII = 64 / WEIGHT_DEPTH;
	N_CIO = 64 / BLK_DEPTH;
	N_COI = 64 / WEIGHT_DEPTH;
	N_COO = 128 / BLK_DEPTH;
	N_SPI = 56 / 7;
	N_SPO = 56 / 7;
	stride = 2;

	off_row = 112 / N_SPO;
	off_col = 112 % N_SPO;
	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += N_BLK;
		weights_all_index += 8;
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 112);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, 0, 0, 56, 2);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += N_BLK;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 56);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 56, 1);
			}
		}
	}

	/////////////////////////////// 128 56 56 ////////////////////////////

	N_CII = 128 / WEIGHT_DEPTH;
	N_CIO = 128 / BLK_DEPTH;
	N_COI = 128 / WEIGHT_DEPTH;
	N_COO = 128 / BLK_DEPTH;
	N_SPI = 56 / 7;
	N_SPO = 56 / 7;
	stride = 1;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += N_BLK;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPI);
		off_col = cio % (112 / N_SPI);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 56);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 56, 1);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 56);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 56, 1);
			}
		}
	}

	/////////////////////////////// 256 28 28 ////////////////////////////

	N_CII = 128 / WEIGHT_DEPTH;
	N_CIO = 128 / BLK_DEPTH;
	N_COI = 128 / WEIGHT_DEPTH;
	N_COO = 256 / BLK_DEPTH;
	N_SPI = 28 / 7;
	N_SPO = 28 / 7;
	stride = 2;
	off_row = 112 / N_SPO;
	off_col = 112 % N_SPO;
	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPI);
		off_col = cio % (112 / N_SPI);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 56);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				int off_row = 112 / N_SPO;
				int off_col = 112 % N_SPO;
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 28, 2);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 28);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				int off_row = 112 / N_SPO;
				int off_col = 112 % N_SPO;
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 28, 1);
			}
		}
	}

	/////////////////////////////// 256 28 28 ////////////////////////////

	N_CII = 256 / WEIGHT_DEPTH;
	N_CIO = 256 / BLK_DEPTH;
	N_COI = 256 / WEIGHT_DEPTH;
	N_COO = 256 / BLK_DEPTH;
	N_SPI = 28 / 7;
	N_SPO = 28 / 7;
	stride = 1;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPI);
		off_col = cio % (112 / N_SPI);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 1, 0, 28);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, cio, off_row, off_col, 28, 1);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPI);
		off_col = coo % (112 / N_SPI);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 28);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 28, 1);
			}
		}
	}

	/////////////////////////////// 512 14 14 ////////////////////////////

	N_CII = 256 / WEIGHT_DEPTH;
	N_CIO = 256 / BLK_DEPTH;
	N_COI = 256 / WEIGHT_DEPTH;
	N_COO = 512 / BLK_DEPTH;
	N_SPI = 14 / 7;
	N_SPO = 14 / 7;
	stride = 2;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPI);
		off_col = cio % (112 / N_SPI);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 28);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 14, 2);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 14, 1);
			}
		}
	}

	/////////////////////////////// 512 14 14 ////////////////////////////

	N_CII = 256 / WEIGHT_DEPTH;
	N_CIO = 256 / BLK_DEPTH;
	N_COI = 256 / WEIGHT_DEPTH;
	N_COO = 512 / BLK_DEPTH;
	N_SPI = 14 / 7;
	N_SPO = 14 / 7;
	stride = 1;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPI);
		off_col = cio % (112 / N_SPI);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 14, 1);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 14, 1);
			}
		}
	}

	/////////////////////////////// 512 14 14 ////////////////////////////

	N_CII = 256 / WEIGHT_DEPTH;
	N_CIO = 256 / BLK_DEPTH;
	N_COI = 256 / WEIGHT_DEPTH;
	N_COO = 512 / BLK_DEPTH;
	N_SPI = 14 / 7;
	N_SPO = 14 / 7;
	stride = 1;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPI);
		off_col = cio % (112 / N_SPI);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 14, 1);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 14, 1);
			}
		}
	}

	/////////////////////////////// 512 14 14 ////////////////////////////

	N_CII = 512 / WEIGHT_DEPTH;
	N_CIO = 512 / BLK_DEPTH;
	N_COI = 512 / WEIGHT_DEPTH;
	N_COO = 512 / BLK_DEPTH;
	N_SPI = 14 / 7;
	N_SPO = 14 / 7;
	stride = 1;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPI);
		off_col = cio % (112 / N_SPI);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 14, 1);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 14, 1);
			}
		}
	}

	/////////////////////////////// 512 14 14 ////////////////////////////

	N_CII = 512 / WEIGHT_DEPTH;
	N_CIO = 512 / BLK_DEPTH;
	N_COI = 512 / WEIGHT_DEPTH;
	N_COO = 512 / BLK_DEPTH;
	N_SPI = 14 / 7;
	N_SPO = 14 / 7;
	stride = 1;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPI);
		off_col = cio % (112 / N_SPI);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 14, 1);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 14, 1);
			}
		}
	}

	/////////////////////////////// 512 14 14 ////////////////////////////

	N_CII = 512 / WEIGHT_DEPTH;
	N_CIO = 512 / BLK_DEPTH;
	N_COI = 512 / WEIGHT_DEPTH;
	N_COO = 512 / BLK_DEPTH;
	N_SPI = 14 / 7;
	N_SPO = 14 / 7;
	stride = 1;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPI);
		off_col = cio % (112 / N_SPI);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 14, 1);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 14, 1);
			}
		}
	}

	/////////////////////////////// 1024 7 7 ////////////////////////////

	N_CII = 512 / WEIGHT_DEPTH;
	N_CIO = 512 / BLK_DEPTH;
	N_COI = 512 / WEIGHT_DEPTH;
	N_COO = 1024 / BLK_DEPTH;
	N_SPI = 7 / 7;
	N_SPO = 7 / 7;
	stride = 2;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPO);
		off_col = cio % (112 / N_SPO);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 14);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 7, 2);
			}
		}
	}

	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 7);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 7, 1);
			}
		}
	}

	/////////////////////////////// 1024 7 7 ////////////////////////////

	N_CII = 1024 / WEIGHT_DEPTH;
	N_CIO = 1024 / BLK_DEPTH;
	N_COI = 1024 / WEIGHT_DEPTH;
	N_COO = 1024 / BLK_DEPTH;
	N_SPI = 7 / 7;
	N_SPO = 7 / 7;
	stride = 1;

	for (int cio = 0; cio < N_CIO; cio++)
	{
		load_weights_3x3_all(conv_weight_3x3_all, weight_3x3_index, weights_all, weights_all_index);
		weight_3x3_index += 4;
		weights_all_index += 8;
		off_row = cio / (112 / N_SPO);
		off_col = cio % (112 / N_SPO);
		for (int row = 0; row < N_SPI; row++)
		{
			for (int col = 0; col < N_SPI; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, row, col, cio);
				for (int cii = 0; cii < N_CII; cii++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 7);
					pgconv64_1bit(pg_buf0, weight_buf_3x3[0], thres_buf[0], FM_buf_acc0, stride);
				}
				store_bufs_organize(DDR_buff_merge, 1, row, col, cio, off_row, off_col, 7, 1);
			}
		}
	}
	for (int coo = 0; coo < N_COO; coo++)
	{
		load_weights_1x1_all(conv_weight_1x1_all, weight_1x1_index, weights_all, weights_all_index);
		weight_1x1_index += 4;
		weights_all_index += 8;
		off_row = coo / (112 / N_SPO);
		off_col = coo % (112 / N_SPO);
		for (int row = 0; row < N_SPO; row++)
		{
			for (int col = 0; col < N_SPO; col++)
			{
				load_buf_from_DDR(DDR_buff_merge, 1, FM_buf0, row, col, coo);
				for (int coi = 0; coi < N_COI; coi++)
				{
					load_buf_from_buf_all(row, col, 0, 0, 7);
					pgconv64_1x1_1bit(pg_buf0, weight_buf_1x1[0], thres_buf[0], FM_buf_acc0);
				}
				int coo_cat = coo;
				if (coo > N_COI)
				{
					coo_cat = coo - N_COI;
				}
				store_bufs_organize(DDR_buff_merge, 0, row, col, coo, off_row, off_col, 7, 1);
			}
		}
	}

	FIX_FM_acc out_buf[16][64];

avgpool:
	for (int c0 = 0; c0 < 16; c0++)
	{
		int coff = c0 * 2;
		load_buf_from_DDR(DDR_buff_merge, 0, FM_buf0, 0, 0, c0);
		for (int col = 0; col < 32; col++)
		{
#pragma HLS unroll
			out_buf[c0][col] = avgpool_7x7(FM_buf0[col]);
		}
		coff += 1;
		load_buf_from_DDR(DDR_buff_merge, 0, FM_buf1, 0, 0, c0);
		for (int col = 0; col < 32; col++)
		{
#pragma HLS unroll
			out_buf[c0][col + 32] = avgpool_7x7(FM_buf1[col]);
		}
	}

	FIX_WT linear_weight_buf[10][64];
#pragma HLS ARRAY_PARTITION variable = linear_weight_buf complete dim = 1
#pragma HLS ARRAY_PARTITION variable = linear_weight_buf complete dim = 2
	FIX_WT linear_bias_buf[10];
#pragma HLS ARRAY_PARTITION variable = linear_bias_buf complete dim = 1

classifier:
	for (int i = 0; i < 100; i++)
	{
		for (int ii = 0; ii < 16; ii++)
		{
			for (int cc = 0; cc < 10; cc++)
			{
				for (int r = 0; r < 2; r++)
				{
#pragma HLS pipeline
					uint512 DATA = 0;
					DATA.range(511, 0) = linear_weight_all[i * 160 + ii * 10 + cc][r].range(511, 0);
					for (int c = 0; c < 32; c++)
					{
#pragma HLS unroll
						linear_weight_buf[cc][r * 32 + c] = DATA.range(WT_RG + c * 16, c * 16);
					}
				}
			}

			uint256 DATA = 0;
			DATA.range(160, 0) = linear_bias_all[i].range(160, 0);
			for (int c = 0; c < 10; c++)
			{
#pragma HLS unroll
				linear_bias_buf[c] = DATA.range(WT_RG + c * 16, c * 16);
			}

			float result[10];
			matmul(out_buf[ii], linear_weight_buf, linear_bias_buf, result);
			for (int j = 0; j < 10; j++)
			{
#pragma HLS pipeline
				out[i * 10 + j] += result[j];
			}
		}
	}

	return;
}
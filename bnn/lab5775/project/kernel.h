#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
typedef ap_int<32> bit32;
typedef ap_uint<32> ubit32;

void test(ap_uint<1> input_image[1][1][16][16], ap_uint<1> w_conv1[16][1][3][3], ap_fixed<32, 22> bn_t1[16][16][16], ap_uint<16> w_conv2[32][1][3][3], ap_fixed<32, 22> bn_t2[32][8][8], ubit32 w_fc1[256][16], ap_fixed<32, 22> b_fc1[256], ubit32 w_fc2[10][8], ap_fixed<32, 22> fc2[1][10], ap_fixed<32, 22> b_fc2[10]);

#endif
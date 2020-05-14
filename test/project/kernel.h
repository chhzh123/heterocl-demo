#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
typedef ap_int<32> bit32;
typedef ap_uint<32> ubit32;

  void test(hls::stream<bit32>& B_channel, hls::stream<bit32>& C_channel);

#endif
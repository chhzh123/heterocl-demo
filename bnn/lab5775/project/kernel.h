#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
typedef ap_int<32> bit32;
typedef ap_uint<32> ubit32;

  void test(hls::stream<ap_uint<1> >& w_fc2_channel, hls::stream<ap_fixed<20, 10> >& b_fc2_channel, hls::stream<ap_fixed<20, 10> >& b_fc1_channel, hls::stream<ap_uint<1> >& w_fc1_channel, hls::stream<ap_fixed<20, 10> >& bn_t2_channel, hls::stream<ap_uint<1> >& w_conv2_channel, hls::stream<ap_fixed<20, 10> >& bn_t1_channel, hls::stream<ap_uint<1> >& w_conv1_channel, hls::stream<ap_uint<1> >& input_image_channel, hls::stream<ap_fixed<20, 10> >& fc2_channel);

#endif
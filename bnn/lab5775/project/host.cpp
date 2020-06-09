
#include <sys/ipc.h>
#include <sys/shm.h>

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <cassert>

// vivado hls headers
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include "kernel.h"

int main(int argc, char ** argv) {
  uint8_t* arg_0 = (uint8_t*)shmat(43909149, nullptr, 0);
  uint8_t* input_image = new uint8_t[1 * 1 * 16 * 16];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        for (size_t i3 = 0; i3 < 16; i3++) {
          input_image[i3 + i2*16 + i1*256 + i0*256] = (uint8_t)(arg_0[i3 + i2*16 + i1*256 + i0*256]);
        }
      }
    }
  }

  uint8_t* arg_1 = (uint8_t*)shmat(43941920, nullptr, 0);
  uint8_t* w_conv1 = new uint8_t[16 * 1 * 3 * 3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          w_conv1[i3 + i2*3 + i1*9 + i0*9] = (uint8_t)(arg_1[i3 + i2*3 + i1*9 + i0*9]);
        }
      }
    }
  }

  int32_t* arg_2 = (int32_t*)shmat(43974692, nullptr, 0);
  int32_t* bn_t1 = new int32_t[16 * 16 * 16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        bn_t1[i2 + i1*16 + i0*256] = (int32_t)(arg_2[i2 + i1*16 + i0*256]) >> 10;
      }
    }
  }

  uint8_t* arg_3 = (uint8_t*)shmat(44007462, nullptr, 0);
  uint8_t* w_conv2 = new uint8_t[32 * 1 * 3 * 3];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          w_conv2[i3 + i2*3 + i1*9 + i0*9] = (uint8_t)(arg_3[i3 + i2*3 + i1*9 + i0*9]);
        }
      }
    }
  }

  int32_t* arg_4 = (int32_t*)shmat(44040232, nullptr, 0);
  int32_t* bn_t2 = new int32_t[32 * 8 * 8];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      for (size_t i2 = 0; i2 < 8; i2++) {
        bn_t2[i2 + i1*8 + i0*64] = (int32_t)(arg_4[i2 + i1*8 + i0*64]) >> 10;
      }
    }
  }

  uint32_t* arg_5 = (uint32_t*)shmat(44073001, nullptr, 0);
  uint32_t* w_fc1 = new uint32_t[256 * 16];
  for (size_t i0 = 0; i0 < 256; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      w_fc1[i1 + i0*16] = (uint32_t)(arg_5[i1 + i0*16]);
    }
  }

  int32_t* arg_6 = (int32_t*)shmat(44105776, nullptr, 0);
  int32_t* b_fc1 = new int32_t[256];
  for (size_t i0 = 0; i0 < 256; i0++) {
    b_fc1[i0] = (int32_t)(arg_6[i0]) >> 10;
  }

  uint32_t* arg_7 = (uint32_t*)shmat(44138547, nullptr, 0);
  uint32_t* w_fc2 = new uint32_t[10 * 8];
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      w_fc2[i1 + i0*8] = (uint32_t)(arg_7[i1 + i0*8]);
    }
  }

  int32_t* arg_8 = (int32_t*)shmat(44171323, nullptr, 0);
  int32_t* b_fc2 = new int32_t[10];
  for (size_t i0 = 0; i0 < 10; i0++) {
    b_fc2[i0] = (int32_t)(arg_8[i0]) >> 10;
  }

  int32_t* arg_9 = (int32_t*)shmat(44204092, nullptr, 0);
  int32_t* fc2 = new int32_t[1 * 10];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      fc2[i1 + i0*10] = (int32_t)(arg_9[i1 + i0*10]) >> 10;
    }
  }


  // compute and kernel call from host
  ap_int<32> _top;
  hls::stream<ap_uint<32> > w_fc1_channel;
  for (ap_int<32> w_fc10 = 0; w_fc10 < 256; ++w_fc10) {
    for (ap_int<32> w_fc11 = 0; w_fc11 < 16; ++w_fc11) {
      w_fc1_channel.write(w_fc1[(w_fc11 + (w_fc10 * 16))]);
    }
  }
  hls::stream<ap_uint<1> > input_image_channel;
  for (ap_int<32> input_image2 = 0; input_image2 < 16; ++input_image2) {
    for (ap_int<32> input_image3 = 0; input_image3 < 16; ++input_image3) {
      input_image_channel.write(input_image[(input_image3 + (input_image2 * 16))]);
    }
  }
  hls::stream<ap_uint<1> > w_conv1_channel;
  for (ap_int<32> w_conv10 = 0; w_conv10 < 16; ++w_conv10) {
    for (ap_int<32> w_conv12 = 0; w_conv12 < 3; ++w_conv12) {
      for (ap_int<32> w_conv13 = 0; w_conv13 < 3; ++w_conv13) {
        w_conv1_channel.write(w_conv1[((w_conv13 + (w_conv12 * 3)) + (w_conv10 * 9))]);
      }
    }
  }
  hls::stream<ap_fixed<20, 10> > b_fc2_channel;
  for (ap_int<32> b_fc20 = 0; b_fc20 < 10; ++b_fc20) {
    b_fc2_channel.write(b_fc2[b_fc20]);
  }
  hls::stream<ap_fixed<20, 10> > bn_t1_channel;
  for (ap_int<32> bn_t10 = 0; bn_t10 < 16; ++bn_t10) {
    for (ap_int<32> bn_t11 = 0; bn_t11 < 16; ++bn_t11) {
      for (ap_int<32> bn_t12 = 0; bn_t12 < 16; ++bn_t12) {
        bn_t1_channel.write(bn_t1[((bn_t12 + (bn_t11 * 16)) + (bn_t10 * 256))]);
      }
    }
  }
  hls::stream<ap_fixed<20, 10> > b_fc1_channel;
  for (ap_int<32> b_fc10 = 0; b_fc10 < 256; ++b_fc10) {
    b_fc1_channel.write(b_fc1[b_fc10]);
  }
  hls::stream<ap_uint<32> > w_fc2_channel;
  for (ap_int<32> w_fc20 = 0; w_fc20 < 10; ++w_fc20) {
    for (ap_int<32> w_fc21 = 0; w_fc21 < 8; ++w_fc21) {
      w_fc2_channel.write(w_fc2[(w_fc21 + (w_fc20 * 8))]);
    }
  }
  hls::stream<ap_fixed<20, 10> > bn_t2_channel;
  for (ap_int<32> bn_t20 = 0; bn_t20 < 32; ++bn_t20) {
    for (ap_int<32> bn_t21 = 0; bn_t21 < 8; ++bn_t21) {
      for (ap_int<32> bn_t22 = 0; bn_t22 < 8; ++bn_t22) {
        bn_t2_channel.write(bn_t2[((bn_t22 + (bn_t21 * 8)) + (bn_t20 * 64))]);
      }
    }
  }
  hls::stream<ap_uint<16> > w_conv2_channel;
  for (ap_int<32> w_conv20 = 0; w_conv20 < 32; ++w_conv20) {
    for (ap_int<32> w_conv22 = 0; w_conv22 < 3; ++w_conv22) {
      for (ap_int<32> w_conv23 = 0; w_conv23 < 3; ++w_conv23) {
        w_conv2_channel.write(w_conv2[((w_conv23 + (w_conv22 * 3)) + (w_conv20 * 9))]);
      }
    }
  }
  hls::stream<ap_fixed<20, 10> > fc2_channel;
  test(w_fc2_channel, b_fc1_channel, w_fc1_channel, bn_t2_channel, w_conv2_channel, bn_t1_channel, w_conv1_channel, input_image_channel, b_fc2_channel, fc2_channel);
  for (ap_int<32> fc21 = 0; fc21 < 10; ++fc21) {
    fc2[fc21] = fc2_channel.read();
  }

  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        for (size_t i3 = 0; i3 < 16; i3++) {
          arg_0[i3 + i2*16 + i1*256 + i0*256] = (uint8_t)(input_image[i3 + i2*16 + i1*256 + i0*256]);
        }
      }
    }
  }
  shmdt(arg_0);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          arg_1[i3 + i2*3 + i1*9 + i0*9] = (uint8_t)(w_conv1[i3 + i2*3 + i1*9 + i0*9]);
        }
      }
    }
  }
  shmdt(arg_1);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        arg_2[i2 + i1*16 + i0*256] = (int32_t)(bn_t1[i2 + i1*16 + i0*256]) << 10;
      }
    }
  }
  shmdt(arg_2);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          arg_3[i3 + i2*3 + i1*9 + i0*9] = (uint8_t)(w_conv2[i3 + i2*3 + i1*9 + i0*9]);
        }
      }
    }
  }
  shmdt(arg_3);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      for (size_t i2 = 0; i2 < 8; i2++) {
        arg_4[i2 + i1*8 + i0*64] = (int32_t)(bn_t2[i2 + i1*8 + i0*64]) << 10;
      }
    }
  }
  shmdt(arg_4);
  for (size_t i0 = 0; i0 < 256; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      arg_5[i1 + i0*16] = (uint32_t)(w_fc1[i1 + i0*16]);
    }
  }
  shmdt(arg_5);
  for (size_t i0 = 0; i0 < 256; i0++) {
    arg_6[i0] = (int32_t)(b_fc1[i0]) << 10;
  }
  shmdt(arg_6);
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      arg_7[i1 + i0*8] = (uint32_t)(w_fc2[i1 + i0*8]);
    }
  }
  shmdt(arg_7);
  for (size_t i0 = 0; i0 < 10; i0++) {
    arg_8[i0] = (int32_t)(b_fc2[i0]) << 10;
  }
  shmdt(arg_8);
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      arg_9[i1 + i0*10] = (int32_t)(fc2[i1 + i0*10]) << 10;
    }
  }
  shmdt(arg_9);


  }

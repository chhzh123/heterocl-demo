
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

#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

int main(int argc, char ** argv) {
  uint8_t* arg_0 = (uint8_t*)shmat(/*input_image*/1048578, nullptr, 0);
  uint8_t input_image[1][1][16][16];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        for (size_t i3 = 0; i3 < 16; i3++) {
          input_image[i0][i1][i2][i3] = (uint8_t)(arg_0[i3 + i2*16 + i1*256 + i0*256]);
        }
      }
    }
  }

  uint8_t* arg_1 = (uint8_t*)shmat(/*w_conv1*/1081347, nullptr, 0);
  uint8_t w_conv1[16][1][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          w_conv1[i0][i1][i2][i3] = (uint8_t)(arg_1[i3 + i2*3 + i1*9 + i0*9]);
        }
      }
    }
  }

  int32_t* arg_2 = (int32_t*)shmat(/*bn_t1*/1114116, nullptr, 0);
  int32_t bn_t1[16][16][16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        bn_t1[i0][i1][i2] = (int32_t)(arg_2[i2 + i1*16 + i0*256]) >> 10;
      }
    }
  }

  uint8_t* arg_3 = (uint8_t*)shmat(/*w_conv2*/1146885, nullptr, 0);
  uint8_t w_conv2[32][1][3][3];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          w_conv2[i0][i1][i2][i3] = (uint8_t)(arg_3[i3 + i2*3 + i1*9 + i0*9]);
        }
      }
    }
  }

  int32_t* arg_4 = (int32_t*)shmat(/*bn_t2*/1179654, nullptr, 0);
  int32_t bn_t2[32][8][8];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      for (size_t i2 = 0; i2 < 8; i2++) {
        bn_t2[i0][i1][i2] = (int32_t)(arg_4[i2 + i1*8 + i0*64]) >> 10;
      }
    }
  }

  uint32_t* arg_5 = (uint32_t*)shmat(/*w_fc1*/1212423, nullptr, 0);
  uint32_t w_fc1[256][16];
  for (size_t i0 = 0; i0 < 256; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      w_fc1[i0][i1] = (uint32_t)(arg_5[i1 + i0*16]);
    }
  }

  int32_t* arg_6 = (int32_t*)shmat(/*b_fc1*/1245192, nullptr, 0);
  int32_t b_fc1[256];
  for (size_t i0 = 0; i0 < 256; i0++) {
    b_fc1[i0] = (int32_t)(arg_6[i0]) >> 10;
  }

  uint32_t* arg_7 = (uint32_t*)shmat(/*w_fc2*/1277961, nullptr, 0);
  uint32_t w_fc2[10][8];
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      w_fc2[i0][i1] = (uint32_t)(arg_7[i1 + i0*8]);
    }
  }

  int32_t* arg_8 = (int32_t*)shmat(/*b_fc2*/1310730, nullptr, 0);
  int32_t b_fc2[10];
  for (size_t i0 = 0; i0 < 10; i0++) {
    b_fc2[i0] = (int32_t)(arg_8[i0]) >> 10;
  }

  int32_t* arg_9 = (int32_t*)shmat(/*fc2*/1343499, nullptr, 0);
  int32_t fc2[1][10];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      fc2[i0][i1] = (int32_t)(arg_9[i1 + i0*10]) >> 10;
    }
  }


  // compute and kernel call from host
  ap_int<32> _top;
  hls::stream<ap_uint<32> > w_fc1_channel;
  for (ap_int<32> w_fc10 = 0; w_fc10 < 256; ++w_fc10) {
    for (ap_int<32> w_fc11 = 0; w_fc11 < 16; ++w_fc11) {
      w_fc1_channel << w_fc1[w_fc10][w_fc11];
    }
  }
  hls::stream<ap_uint<1> > input_image_channel;
  for (ap_int<32> input_image2 = 0; input_image2 < 16; ++input_image2) {
    for (ap_int<32> input_image3 = 0; input_image3 < 16; ++input_image3) {
      input_image_channel << input_image[0][0][input_image2][input_image3];
    }
  }
  hls::stream<ap_uint<1> > w_conv1_channel;
  for (ap_int<32> w_conv10 = 0; w_conv10 < 16; ++w_conv10) {
    for (ap_int<32> w_conv12 = 0; w_conv12 < 3; ++w_conv12) {
      for (ap_int<32> w_conv13 = 0; w_conv13 < 3; ++w_conv13) {
        w_conv1_channel << w_conv1[w_conv10][0][w_conv12][w_conv13];
      }
    }
  }
  hls::stream<ap_fixed<20, 10> > b_fc2_channel;
  for (ap_int<32> b_fc20 = 0; b_fc20 < 10; ++b_fc20) {
    b_fc2_channel << b_fc2[b_fc20];
  }
  hls::stream<ap_fixed<20, 10> > bn_t1_channel;
  for (ap_int<32> bn_t10 = 0; bn_t10 < 16; ++bn_t10) {
    for (ap_int<32> bn_t11 = 0; bn_t11 < 16; ++bn_t11) {
      for (ap_int<32> bn_t12 = 0; bn_t12 < 16; ++bn_t12) {
        bn_t1_channel << bn_t1[bn_t10][bn_t11][bn_t12];
      }
    }
  }
  hls::stream<ap_fixed<20, 10> > b_fc1_channel;
  for (ap_int<32> b_fc10 = 0; b_fc10 < 256; ++b_fc10) {
    b_fc1_channel << b_fc1[b_fc10];
  }
  hls::stream<ap_uint<32> > w_fc2_channel;
  for (ap_int<32> w_fc20 = 0; w_fc20 < 10; ++w_fc20) {
    for (ap_int<32> w_fc21 = 0; w_fc21 < 8; ++w_fc21) {
      w_fc2_channel << w_fc2[w_fc20][w_fc21];
    }
  }
  hls::stream<ap_fixed<20, 10> > bn_t2_channel;
  for (ap_int<32> bn_t20 = 0; bn_t20 < 32; ++bn_t20) {
    for (ap_int<32> bn_t21 = 0; bn_t21 < 8; ++bn_t21) {
      for (ap_int<32> bn_t22 = 0; bn_t22 < 8; ++bn_t22) {
        bn_t2_channel << bn_t2[bn_t20][bn_t21][bn_t22];
      }
    }
  }
  hls::stream<ap_uint<16> > w_conv2_channel;
  for (ap_int<32> w_conv20 = 0; w_conv20 < 32; ++w_conv20) {
    for (ap_int<32> w_conv22 = 0; w_conv22 < 3; ++w_conv22) {
      for (ap_int<32> w_conv23 = 0; w_conv23 < 3; ++w_conv23) {
        w_conv2_channel << w_conv2[w_conv20][0][w_conv22][w_conv23];
      }
    }
  }
  hls::stream<ap_fixed<20, 10> > fc2_channel;
  test(b_fc1_channel, w_fc1_channel, w_conv2_channel, bn_t1_channel, w_conv1_channel, input_image_channel, bn_t2_channel, w_fc2_channel, b_fc2_channel, fc2_channel);
  for (ap_int<32> fc21 = 0; fc21 < 10; ++fc21) {
    fc2[0][fc21] = fc2_channel.read();
  }

  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        for (size_t i3 = 0; i3 < 16; i3++) {
          arg_0[i3 + i2*16 + i1*256 + i0*256] = (uint8_t)(input_image[i0][i1][i2][i3]);
        }
      }
    }
  }
  shmdt(arg_0);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          arg_1[i3 + i2*3 + i1*9 + i0*9] = (uint8_t)(w_conv1[i0][i1][i2][i3]);
        }
      }
    }
  }
  shmdt(arg_1);
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        arg_2[i2 + i1*16 + i0*256] = (int32_t)(bn_t1[i0][i1][i2]) << 10;
      }
    }
  }
  shmdt(arg_2);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          arg_3[i3 + i2*3 + i1*9 + i0*9] = (uint8_t)(w_conv2[i0][i1][i2][i3]);
        }
      }
    }
  }
  shmdt(arg_3);
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      for (size_t i2 = 0; i2 < 8; i2++) {
        arg_4[i2 + i1*8 + i0*64] = (int32_t)(bn_t2[i0][i1][i2]) << 10;
      }
    }
  }
  shmdt(arg_4);
  for (size_t i0 = 0; i0 < 256; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      arg_5[i1 + i0*16] = (uint32_t)(w_fc1[i0][i1]);
    }
  }
  shmdt(arg_5);
  for (size_t i0 = 0; i0 < 256; i0++) {
    arg_6[i0] = (int32_t)(b_fc1[i0]) << 10;
  }
  shmdt(arg_6);
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      arg_7[i1 + i0*8] = (uint32_t)(w_fc2[i0][i1]);
    }
  }
  shmdt(arg_7);
  for (size_t i0 = 0; i0 < 10; i0++) {
    arg_8[i0] = (int32_t)(b_fc2[i0]) << 10;
  }
  shmdt(arg_8);
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      arg_9[i1 + i0*10] = (int32_t)(fc2[i0][i1]) << 10;
    }
  }
  shmdt(arg_9);


  }

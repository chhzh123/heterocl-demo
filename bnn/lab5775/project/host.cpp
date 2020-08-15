
#include <sys/ipc.h>
#include <sys/shm.h>

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <iostream>
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
#include <hls_math.h>
#include <math.h>
#include <stdint.h>

int main(int argc, char ** argv) {
  std::cout << " Initialize shared memory...";
  uint8_t* arg_0 = (uint8_t*)shmat(/*input_image*/57278509, nullptr, 0);
  auto input_image = new ap_uint<1>[1][1][16][16];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        for (size_t i3 = 0; i3 < 16; i3++) {
          input_image[i0][i1][i2][i3] = (uint8_t)(arg_0[i3 + i2*16 + i1*256 + i0*256]);
        }
      }
    }
  }

  uint8_t* arg_1 = (uint8_t*)shmat(/*w_conv1*/57311301, nullptr, 0);
  auto w_conv1 = new ap_uint<1>[16][1][3][3];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          w_conv1[i0][i1][i2][i3] = (uint8_t)(arg_1[i3 + i2*3 + i1*9 + i0*9]);
        }
      }
    }
  }

  int32_t* arg_2 = (int32_t*)shmat(/*bn_t1*/57344086, nullptr, 0);
  auto bn_t1 = new ap_fixed<32,22>[16][16][16];
  for (size_t i0 = 0; i0 < 16; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      for (size_t i2 = 0; i2 < 16; i2++) {
        bn_t1[i0][i1][i2] = (int32_t)(arg_2[i2 + i1*16 + i0*256]) >> 10;
      }
    }
  }

  uint16_t* arg_3 = (uint16_t*)shmat(/*w_conv2*/57376855, nullptr, 0);
  auto w_conv2 = new ap_uint<16>[32][1][3][3];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 1; i1++) {
      for (size_t i2 = 0; i2 < 3; i2++) {
        for (size_t i3 = 0; i3 < 3; i3++) {
          w_conv2[i0][i1][i2][i3] = (uint16_t)(arg_3[i3 + i2*3 + i1*9 + i0*9]);
        }
      }
    }
  }

  int32_t* arg_4 = (int32_t*)shmat(/*bn_t2*/57409624, nullptr, 0);
  auto bn_t2 = new ap_fixed<32,22>[32][8][8];
  for (size_t i0 = 0; i0 < 32; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      for (size_t i2 = 0; i2 < 8; i2++) {
        bn_t2[i0][i1][i2] = (int32_t)(arg_4[i2 + i1*8 + i0*64]) >> 10;
      }
    }
  }

  uint32_t* arg_5 = (uint32_t*)shmat(/*w_fc1*/57442393, nullptr, 0);
  auto w_fc1 = new ap_uint<32>[256][16];
  for (size_t i0 = 0; i0 < 256; i0++) {
    for (size_t i1 = 0; i1 < 16; i1++) {
      w_fc1[i0][i1] = (uint32_t)(arg_5[i1 + i0*16]);
    }
  }

  int32_t* arg_6 = (int32_t*)shmat(/*b_fc1*/57475162, nullptr, 0);
  auto b_fc1 = new ap_fixed<32,22>[256];
  for (size_t i0 = 0; i0 < 256; i0++) {
    b_fc1[i0] = (int32_t)(arg_6[i0]) >> 10;
  }

  uint32_t* arg_7 = (uint32_t*)shmat(/*w_fc2*/57507931, nullptr, 0);
  auto w_fc2 = new ap_uint<32>[10][8];
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 8; i1++) {
      w_fc2[i0][i1] = (uint32_t)(arg_7[i1 + i0*8]);
    }
  }

  int32_t* arg_8 = (int32_t*)shmat(/*b_fc2*/57540700, nullptr, 0);
  auto b_fc2 = new ap_fixed<32,22>[10];
  for (size_t i0 = 0; i0 < 10; i0++) {
    b_fc2[i0] = (int32_t)(arg_8[i0]) >> 10;
  }

  int32_t* arg_9 = (int32_t*)shmat(/*fc2*/57573469, nullptr, 0);
  auto fc2 = new ap_fixed<32,22>[1][10];
  for (size_t i0 = 0; i0 < 1; i0++) {
    for (size_t i1 = 0; i1 < 10; i1++) {
      fc2[i0][i1] = (int32_t)(arg_9[i1 + i0*10]) >> 10;
    }
  }

  std::cout << " Initialize RTE...";

  // compute and kernel call from host
  ap_int<32> _top;
  test(input_image, w_conv1, bn_t1, w_conv2, bn_t2, w_fc1, b_fc1, w_fc2, fc2, b_fc2);

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
          arg_3[i3 + i2*3 + i1*9 + i0*9] = (uint16_t)(w_conv2[i0][i1][i2][i3]);
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

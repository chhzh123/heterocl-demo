#ifndef __SIMPLE_ADD_H__
#define __SIMPLE_ADD_H__

#include <ap_int.h>
#include <ap_fixed.h>
#include <math.h>
using namespace std;

#define HW_COSIM

#define A_ROWS 10
#define A_COLS 10

typedef ap_int<32> in_data_t;
typedef ap_uint<4> out_data_t;

void default_function(in_data_t A[10][10], out_data_t B[8][8]);

#endif // SIMPLE_ADD_H
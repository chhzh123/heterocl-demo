#include "simple_add.h"
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
	in_data_t A[10][10] = {{9,3,8,9,7,5,5,1,1,0},
                           {3,3,3,1,6,3,1,5,1,4},
                           {4,2,0,0,1,1,6,1,5,5},
                           {2,0,4,8,3,0,7,5,9,2},
                           {5,2,1,4,9,6,4,0,1,7},
                           {2,9,4,4,2,6,6,0,4,6},
                           {6,5,2,1,9,2,9,8,5,1},
                           {9,9,2,3,7,4,3,5,9,5},
                           {5,6,6,3,6,8,0,1,5,3},
                           {8,4,2,0,5,8,4,0,9,9}};
    out_data_t hw_result[8][8], sw_result[8][8];
    int error_count = 0;

    // Generate the expected result
    for (int i = 0; i < A_ROWS-2; ++i)
        for (int j = 0; j < A_COLS-2; ++j)
            sw_result[i][j] = A[i][j] + A[i+2][j+2];

  in_data_t A_stream[100];
  out_data_t hw_result_stream[64];
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      A_stream[i * 10 + j] = A[i][j];
#ifdef HW_COSIM
    // Run the Vivado HLS matrix adder
    default_function(A_stream, hw_result_stream);
#endif

    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        hw_result[i][j] = hw_result_stream[i * 8 + j];

    // Print results
    for (int i = 0; i < A_ROWS-2; i++)
    {
        for (int j = 0; j < A_COLS-2; j++)
        {
#ifdef HW_COSIM
            // Check result of HLS vs. expected
            if (hw_result[i][j] != sw_result[i][j])
            {
                error_count++;
            }
#else
            cout << sw_result[i][j] << " ";
#endif
        }
    }

#ifdef HW_COSIM
    if (error_count)
        cout << "TEST FAIL: " << error_count << "Results do not match!" << endl;
    else
        cout << "Test passed!" << endl;
#endif
    return error_count;
}
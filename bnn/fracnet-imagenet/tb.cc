
#include "net_hls.h"
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>
//#include "weights_dummy.h"

float thermo_image_g[96][32][32];

float conv1_weight_g[16][96][3][3];
float conv1_output_g[16][32][32];

float bn1_weight_g[16];
float bn1_bias_g[16];
float bn1_output_g[16][32][32];

uint16 image_thermo[6*224*224];

uint512 conv_weight_1x1_all[1000];
uint512 conv_weight_3x3_all[1000][3][3];
uint512 weights_all[10000];
uint512 linear_weight_all[16000][2];
uint512 linear_bias_all[100];

uint512 DDR_buff_merge[861184];

float out[1000];

void golden_model();

void load_weights()
{
    std::ifstream ifs_param0("image_thermo.bin", std::ios::in | std::ios::binary);
    ifs_param0.read((char*)(image_thermo), 6*224*224*sizeof(uint16));
    ifs_param0.close();
    std::ifstream ifs_param1("conv_weight_1x1_all.bin", std::ios::in | std::ios::binary);
    ifs_param1.read((char*)(conv_weight_1x1_all), 16*sizeof(uint512));
    ifs_param1.close();
    std::ifstream ifs_param2("conv_weight_3x3_all.bin", std::ios::in | std::ios::binary);
    ifs_param2.read((char*)(**conv_weight_3x3_all), 1000*3*3*sizeof(uint512));
    ifs_param2.close();
    std::ifstream ifs_param3("weights_all.bin", std::ios::in | std::ios::binary);
    ifs_param3.read((char*)(weights_all), 10000*sizeof(uint512));
    ifs_param3.close();
    std::ifstream ifs_param4("linear_weight_all.bin", std::ios::in | std::ios::binary);
    ifs_param4.read((char*)(*linear_weight_all), 32000*sizeof(uint512));
    ifs_param4.close();
    std::ifstream ifs_param5("linear_bias_all.bin", std::ios::in | std::ios::binary);
    ifs_param5.read((char*)(linear_bias_all), 100*sizeof(uint512));
    ifs_param5.close();
}

int test( char* img)
{
    // Read Image 96*32*32 bytes
    // std::ifstream ifs_image_raw(img, std::ios::in | std::ios::binary);
    // ifs_image_raw.read((char*)(**image_raw_g), 96*32*32*sizeof(uint8));
    // ifs_image_raw.close();

    // for (int c = 0; c < 96; c ++) {
    //     for (int row = 0; row < 32; row ++) {
    //         for (int col = 0; col < 32; col ++) {
    //          thermo_image_g[c][row][col] = image_raw_g[c][row][col].to_int()/1.0;
    //         }
    //     }
    // }

    // Golden Model
//    golden_model();
    FracNet(image_thermo,
            conv_weight_1x1_all,
            conv_weight_3x3_all,
            weights_all,
            linear_weight_all,
            linear_bias_all,
            DDR_buff_merge,
            out
);
    std::cout << out;
return 1;
}

int main()
{

    load_weights();

    printf("Testing on Image\n");
    FracNet(image_thermo,
                conv_weight_1x1_all,
                conv_weight_3x3_all,
                weights_all,
                linear_weight_all,
                linear_bias_all,
                DDR_buff_merge,
                out
    );
    std::cout << out;
    printf("Tested\n");
    return 0;
}

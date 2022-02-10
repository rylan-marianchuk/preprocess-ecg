#include <iostream>
#include <vector>
#include <thrust/sort.h>
#include <cstdlib>
#include <algorithm>

#include "cudaResults.h"

__device__ bool hasflat20samples(float * signal, size_t NCOL){
    float prev = signal[0];
    short int longest = 0;
    for (size_t i = 1; i < NCOL; i++){
        if (signal[i] == prev)
            longest++;
        else
            longest = 0;

        prev = signal[i];

        if (longest == 20)
            return 1;
    }
    return 0;
}

__device__ float histentropy(float * signal, size_t NCOL, int bins=40){

    thrust::sort(thrust::seq, signal, signal + NCOL);
    float min = signal[0];
    float max = signal[NCOL - 1];
    float sum = 0;

    float binSize = (max - min) / bins;
    float binCount = 0;
    for (size_t i = 0; i < NCOL; i++){
        if (signal[i] <= min + binSize){
            binCount++;
        }
        else {
            float v = binCount / binSize / NCOL;
            sum += std::log2(v) * (v);
            binCount = 1;
            min += binSize;
        }
    }
    float v = binCount / binSize / NCOL;
    sum += std::log2(v) * (v);

    return -sum;
}

__device__ float curvelength(float * signal, size_t NCOL){
    float CL = 0;
    for (int j = 0; j < NCOL - 1; j++) {
        float x1 = signal[j];
        float x2 = signal[j+1];
        CL += std::sqrt(1.0f + (x2 - x1)*(x2 - x1));
    }
    return CL;
}

__global__ void GetParamsGPU(float* ecgs, const size_t SIGNALS, const size_t NCOL,
                            bool * flat20_res,
                            float * curve_length_res,
                            float * histentropy_res) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    while (i < SIGNALS){
        float * signal = ecgs + (i * NCOL);

        flat20_res[i] = 0;
        curve_length_res[i] = 0.0f;
        histentropy_res[i] = 0.0f;

        // 20 values equal in sequence?
        flat20_res[i] = hasflat20samples(signal, NCOL);

        // Curve Length
        curve_length_res[i] = curvelength(signal, NCOL);

        // Histogram Entropy
        histentropy_res[i] = histentropy(signal, NCOL);



        //printf("Start: \t %.0f %.0f %.0f \t\tEnd: %.0f %.0f %.0f \t\t\t\t\t CL: %f \t\t HE: %f \n", signal[0], signal[1], signal[2], signal[4997],
        //       signal[4998], signal[4999], curve_length_res[i], histentropy_res[i]);

        //printf("Start: \t %.0f %.0f %.0f \t\tEnd: %.0f %.0f %.0f\n", signal[0], signal[1], signal[2], signal[4997],
        //       signal[4998], signal[4999]);

        i += stride;
    }


}

cudaResults getArtifactParams(float * ecgs, const size_t SIGNALS, const size_t NCOL){

    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL*1024);

    cudaResults resP;

    size_t byte_amount_res = sizeof(float) * SIGNALS;

    resP.res20flat = (bool *)malloc(sizeof(bool) * SIGNALS );
    resP.resCL = (float*)malloc(byte_amount_res);
    resP.resHE = (float*)malloc(byte_amount_res);

    bool* d_res20flat;
    float* d_resCL;
    float* d_resHE;
    cudaMalloc(&d_res20flat, sizeof(bool) * SIGNALS );
    cudaMalloc(&d_resCL, byte_amount_res);
    cudaMalloc(&d_resHE, byte_amount_res);

    float * d_ecgs;
    cudaMalloc(&d_ecgs, sizeof(float) * SIGNALS * NCOL);
    cudaMemcpy(d_ecgs, ecgs, sizeof(float) * SIGNALS * NCOL, cudaMemcpyHostToDevice);

    const unsigned tpb_x = 256;
    const unsigned bpg_x = (SIGNALS + tpb_x - 1) / tpb_x;
    dim3 blocksPerGrid(bpg_x, 1, 1);
    dim3 threadsPerBlock(tpb_x, 1, 1);

    GetParamsGPU<<<blocksPerGrid, threadsPerBlock>>>(d_ecgs, SIGNALS, NCOL, d_res20flat,
                                                     d_resCL, d_resHE);

    // Wait for gpu to finish
    cudaDeviceSynchronize();

    cudaMemcpy(resP.res20flat, d_res20flat, sizeof(bool) * SIGNALS, cudaMemcpyDeviceToHost);
    cudaMemcpy(resP.resCL, d_resCL, byte_amount_res, cudaMemcpyDeviceToHost);
    cudaMemcpy(resP.resHE, d_resHE, byte_amount_res, cudaMemcpyDeviceToHost);

    cudaFree(d_res20flat);
    cudaFree(d_resCL);
    cudaFree(d_resHE);
    cudaFree(d_ecgs);


    return resP;
}

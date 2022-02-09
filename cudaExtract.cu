#include <iostream>
#include <vector>
#include <thrust/sort.h>
#include <cstdlib>
#include "cudaResults.h"

__device__ float histentropy(float * signal, size_t NCOL, int bins=40){

    thrust::sort(thrust::seq, signal, signal + NCOL);
    float min = signal[0];
    float max = signal[NCOL - 1];
    float sum = 0;

    float binSize = (max - min) / bins;
    float binCount = 0;
    //printf("Bin: [%f  %f):", min, min+binSize);
    for (size_t i = 0; i < NCOL; i++){
        if (signal[i] <= min + binSize){
            binCount++;
            //printf(" 1 ");
        }
        else {
            float v = binCount / binSize / NCOL;
            sum += std::log2(v) * (v);
            //printf("\n val: %f \n", binCount);

            binCount = 1;
            min += binSize;
            //printf("\nBin: [%f  %f):", min, min+binSize);
            //printf(" 1 ");
        }
    }
    float v = binCount / binSize / NCOL;
    sum += std::log2(v) * (v);

    return -sum;
}


__global__ void CurveLength(float* ecgs, const size_t SIGNALS, const size_t NCOL,
                            float * curve_length_res,
                            float * histentropy_res) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    while (i < SIGNALS){
        float * signal = ecgs + i * NCOL;
        curve_length_res[i] = 0.0f;
        histentropy_res[i] = 0.0f;

        // Curve Length
        for (int j = 0; j < NCOL - 1; j++) {
            float x1 = signal[j];
            float x2 = signal[j+1];
            curve_length_res[i] += std::sqrt(1.0f + (x2 - x1)*(x2 - x1));
        }

        // Histogram Entropy
        histentropy_res[i] = histentropy(signal, NCOL);

        i += stride;
    }


}

cudaResults getArtifactParams(float * ecgs, dim3 blocksPerGrid, dim3 threadsPerBlock, const size_t SIGNALS, const size_t NCOL){

    struct cudaResults resP;

    size_t byte_amount_res = sizeof(float) * SIGNALS;

    resP.resCL = (float*)malloc(byte_amount_res);
    resP.resHE = (float*)malloc(byte_amount_res);

    float* d_resCL;
    float* d_resHE;
    cudaMalloc(&d_resCL, byte_amount_res);
    cudaMalloc(&d_resHE, byte_amount_res);

    float * d_ecgs;
    cudaMalloc(&d_ecgs, sizeof(float) * SIGNALS * NCOL);
    cudaMemcpy(d_ecgs, ecgs, sizeof(float) * SIGNALS * NCOL, cudaMemcpyHostToDevice);


    CurveLength<<<blocksPerGrid, threadsPerBlock>>>(d_ecgs, SIGNALS, NCOL, d_resCL, d_resHE);

    // Wait for gpu to finish
    cudaDeviceSynchronize();

    cudaMemcpy(resP.resCL, d_resCL, byte_amount_res, cudaMemcpyDeviceToHost);
    cudaMemcpy(resP.resHE, d_resHE, byte_amount_res, cudaMemcpyDeviceToHost);


    cudaFree(d_resCL);
    cudaFree(d_resHE);
    cudaFree(d_ecgs);


    return resP;
}

#define SIGNALS 40000
#define NUM_COLS 5000

int run(){
    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL*1024);
    bool showOutput = true;

    float * ecgs = new float[SIGNALS*NUM_COLS];

    for (size_t i = 0; i < SIGNALS; i++){
        for (size_t j = 0; j < NUM_COLS; j++){
            //ecgs[i * NUM_COLS + j] = (float) (i*(NUM_COLS) + j);
            ecgs[i * NUM_COLS + j] = (float) (rand() % 1000);
        }
    }


    const unsigned tpb_x = 256;
    const unsigned bpg_x = (SIGNALS + tpb_x - 1) / tpb_x;
    dim3 blocksperGrid(bpg_x, 1, 1);
    dim3 threadsPerBlock(tpb_x, 1, 1);

    cudaResults res = getArtifactParams(ecgs, blocksperGrid, threadsPerBlock, SIGNALS, NUM_COLS);

    if (showOutput){
        std::cout << "CL" << "\t\t\t" << "HE" << std::endl;
        for (size_t i = 0; i < SIGNALS; i++){
            std::cout << res.resCL[i] << "\t\t" << res.resHE[i] << std::endl;
        }
    }

    return 0;

}

#include <iostream>
#include <vector>
#include <thrust/sort.h>
#include <cstdlib>
#include <chrono>

#include "cudaResults.h"

__device__ int hasflat20samples(float * signal, size_t NCOL){
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

__device__ double histentropy(float * signal, size_t NCOL, int bins=40){

    thrust::sort(thrust::seq, signal, signal + NCOL);
    double min = signal[0];
    double max = signal[NCOL - 1];
    double sum = 0;

    double binSize = (max - min) / bins;
    double binCount = 0;
    for (size_t i = 0; i < NCOL; i++){
        if (signal[i] <= min + binSize){
            binCount++;
        }
        else {
            double v = binCount / binSize / NCOL;
            sum += std::log2(v) * (v);
            binCount = 1;
            min += binSize;
        }
    }
    double v = binCount / binSize / NCOL;
    sum += std::log2(v) * (v);

    return -sum;
}

__device__ double curvelength(float * signal, size_t NCOL){
    double CL = 0;
    for (int j = 0; j < NCOL - 1; j++) {
        double x1 = signal[j];
        double x2 = signal[j+1];
        CL += std::sqrt(1.0f + (x2 - x1)*(x2 - x1));
    }
    return CL;
}

__device__ float segmentautocorrsim(float * signal, size_t NCOL, int seg_size=1250, int nlags=50){
    return 0.0f;
}


__global__ void GetParamsGPU(float * ecgs, const size_t SIGNALS, const size_t NCOL,
                            int * flat20_res,
                            double * curve_length_res,
                            double * histentropy_res) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    while (i < SIGNALS){
        float * signal = ecgs + (i * NCOL);

        flat20_res[i] = 0;
        curve_length_res[i] = 0.0;
        histentropy_res[i] = 0.0;

        // 20 values equal in sequence?
        flat20_res[i] = hasflat20samples(signal, NCOL);

        // Curve Length
        curve_length_res[i] = curvelength(signal, NCOL);

        // Histogram Entropy
        histentropy_res[i] = histentropy(signal, NCOL);

        i += stride;
    }


}

cudaResults getArtifactParams(float * ecgs, const size_t SIGNALS, const size_t NCOL){

    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL*1024);

    cudaResults resP;

    size_t byte_amount_res = sizeof(double) * SIGNALS;

    resP.resCL = (double*)malloc(byte_amount_res);
    resP.resHE = (double*)malloc(byte_amount_res);
    resP.res20flat = (int *)malloc(sizeof(int) * SIGNALS );



    double * d_resCL;
    double * d_resHE;
    int * d_res20flat;


    cudaMalloc(&d_resCL, byte_amount_res);
    cudaMalloc(&d_resHE, byte_amount_res);
    cudaMalloc(&d_res20flat, sizeof(int) * SIGNALS );

    float * d_ecgs;
    cudaMalloc(&d_ecgs, sizeof(float) * SIGNALS * NCOL);
    cudaMemcpy(d_ecgs, ecgs, sizeof(float) * SIGNALS * NCOL, cudaMemcpyHostToDevice);

    const unsigned tpb_x = 256;
    const unsigned bpg_x = (SIGNALS + tpb_x - 1) / tpb_x;
    dim3 blocksPerGrid(bpg_x, 1, 1);
    dim3 threadsPerBlock(tpb_x, 1, 1);
    auto start = std::chrono::steady_clock::now();
    GetParamsGPU<<<blocksPerGrid, threadsPerBlock>>>(d_ecgs, SIGNALS, NCOL, d_res20flat,
                                                     d_resCL, d_resHE);

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << duration.count() << std::endl;

    // Wait for gpu to finish
    cudaDeviceSynchronize();
    std::cout << "Kernel Done" << std::endl;

    cudaMemcpy(resP.resCL, d_resCL, byte_amount_res, cudaMemcpyDeviceToHost);
    cudaMemcpy(resP.resHE, d_resHE, byte_amount_res, cudaMemcpyDeviceToHost);
    cudaMemcpy(resP.res20flat, d_res20flat, sizeof(int) * SIGNALS, cudaMemcpyDeviceToHost);

    cudaFree(d_resCL);
    cudaFree(d_resHE);
    cudaFree(d_res20flat);
    cudaFree(d_ecgs);

    return resP;
}

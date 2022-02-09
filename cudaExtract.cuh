#include "cudaResults.h"

__device__ float histentropy(float * signal, size_t NCOL, int bins=40);

__global__ void CurveLength(float* ecgs, const size_t SIGNALS, const size_t NCOL,
                            float * curve_length_res,
                            float * histentropy_res);

cudaResults getArtifactParams(float * ecgs, dim3 blocksPerGrid, dim3 threadsPerBlock, const size_t SIGNALS, const size_t NCOL);

int run();


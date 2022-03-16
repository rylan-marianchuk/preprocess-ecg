#ifndef CUDAEXTRACT_CUH
#define CUDAEXTRACT_CUH
#include "cudaResults.h"

//__device__ float histentropy(float * signal, size_t NCOL, int bins=40);

//__global__ void CurveLength(float* ecgs, const size_t SIGNALS, const size_t NCOL,
//                            float * curve_length_res,
//                            float * histentropy_res);

cudaResults getArtifactParams(float * ecgs, const size_t SIGNALS, const size_t NCOL);

#endif /* CUDAEXTRACT_CUH */

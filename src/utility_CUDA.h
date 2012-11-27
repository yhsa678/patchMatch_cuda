#ifndef UTILITY_CUDA_H
#define UTILITY_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <string>

#define CUDA_SAFE_CALL(err) _CUDA_SAFE_CALL( err,__FILE__, __LINE__)

void _CUDA_SAFE_CALL( cudaError_t err, std::string file = __FILE__, int line = __LINE__);


#endif
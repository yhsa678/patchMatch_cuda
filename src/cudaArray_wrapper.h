#ifndef CUDAARRAY_WRAPPER_H
#define CUDAARRAY_WRAPPER_H
#include <cuda_runtime.h>
#include "utility_CUDA.h"

class CudaArray_wrapper
{
	cudaArray *_array3D;
	int _width;
	int _height;
	int _depth;

	CudaArray_wrapper(int width, int height, int depth);

	void array3DCopy(char *img,  enum cudaMemcpyKind kind);


};



#endif
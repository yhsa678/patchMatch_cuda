#include "cudaArray_wrapper.h"

CudaArray_wrapper::CudaArray_wrapper(int width, int height, int depth):_width(width), _height(height), _depth(depth), _array3D(NULL)
{
}

CudaArray_wrapper::~CudaArray_wrapper()
{
	if(_array3D != NULL )
		cudaFreeArray(_array3D);
}





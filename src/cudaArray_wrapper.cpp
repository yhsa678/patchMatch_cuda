#include "cudaArray_wrapper.h"

CudaArray_wrapper::CudaArray_wrapper(int width, int height, int depth):_width(width), _height(height), _depth(depth), _array3D(NULL)
{
}

CudaArray_wrapper::~CudaArray_wrapper()
{
	if(_array3D != NULL )
		cudaFreeArray(_array3D);
}

void CudaArray_wrapper::array3DCopy_float(float *data, enum cudaMemcpyKind kind, int dataPitch)
{
	if(kind == cudaMemcpyDeviceToDevice)
	{
		struct cudaMemcpy3DParms params = {0};
		params.extent = make_cudaExtent(_width, _height, _depth);
		params.kind = kind;
		params.dstArray = _array3D;
		params.srcPtr = make_cudaPitchedPtr((void*)data, dataPitch, _width, _height);
		CUDA_SAFE_CALL(cudaMemcpy3D(&params));
	}
}






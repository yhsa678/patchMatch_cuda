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
	if(_array3D == NULL)
	{
		struct cudaExtent extent = make_cudaExtent(_width, _height, _depth);	
		cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();				
		CUDA_SAFE_CALL(cudaMalloc3DArray(&_array3D,&fmt,extent, cudaArrayLayered));	
	}	
	
	if(kind == cudaMemcpyDeviceToDevice  )
	{
		struct cudaMemcpy3DParms parameters = {0};
		parameters.extent = make_cudaExtent(_width, _height, _depth);
		parameters.kind = kind;
		parameters.dstArray = _array3D;
		parameters.srcPtr = make_cudaPitchedPtr((void*)data, dataPitch, _width, _height);
		CUDA_SAFE_CALL(cudaMemcpy3D(&parameters));
	}
	else if(kind == cudaMemcpyHostToDevice)
	{	
		struct cudaMemcpy3DParms parameters = {0};
		parameters.extent = make_cudaExtent(_width, _height, _depth);
		parameters.kind = kind;
		parameters.dstArray = _array3D;
		parameters.srcPtr = make_cudaPitchedPtr((void*)data, _width*sizeof(float), _width, _height);
		CUDA_SAFE_CALL(cudaMemcpy3D(&parameters));
	}
	
}






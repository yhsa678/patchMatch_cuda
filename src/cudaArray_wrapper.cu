#include "cudaArray_wrapper.h"




CudaArray_wrapper::CudaArray_wrapper(int width, int height, int depth)
{
	_width = width;
	_height = height;
	_depth = depth;

// Allocate depthmaps array.
	cudaChannelFormatDesc fmt = cudaCreateChannelDesc<char>();	
	struct cudaExtent extent = make_cudaExtent(_width, _height, _depth);	
	CUDA_SAFE_CALL(cudaMalloc3DArray(&_array3D,&fmt,extent, cudaArrayLayered));	

}

void CudaArray_wrapper::array3DCopy(char *img,  enum cudaMemcpyKind kind)	// It must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice
{
	if(kind == cudaMemcpyHostToDevice)
	{
		struct cudaExtent extent = make_cudaExtent(_width, _height, _depth); //If a CUDA array is participating in the copy, the extent is defined in terms of that array's elements	
		//copy data
		struct cudaMemcpy3DParms params = {0};
		params.extent = extent;
		params.kind = /*cudaMemcpyHostToDevice*/ kind; 
		params.dstArray = _array3D;
		params.srcPtr = make_cudaPitchedPtr((void*)img,_width*sizeof(char),_width,_height);
		CUDA_SAFE_CALL(cudaMemcpy3D(&params));
		//cudaChannelFormatDesc fmt1 = cudaCreateChannelDesc<float>();
		//CUDA_SAFE_CALL(cudaBindTextureToArray(depthmapsTex,_array3D,fmt));	// bind to texture
		//depthmapsTex.normalized = true;
	}
	else if(kind == cudaMemcpyDeviceToHost)
	{ 
		struct cudaMemcpy3DParms params = {0};
		params.extent = make_cudaExtent(_width, _height, _depth);
		params.kind = kind; 
		params.dstPtr = make_cudaPitchedPtr((void*)img,_width*sizeof(char),_width,_height);
		params.srcArray = _array3D;
		CUDA_SAFE_CALL(cudaMemcpy3D(&params));
	}
}
#ifndef CUDAARRAY_WRAPPER_H
#define CUDAARRAY_WRAPPER_H
#include <cuda_runtime.h>
#include "utility_CUDA.h"

class CudaArray_wrapper
{ 	
	int _width;
	int _height;
	int _depth;

	//CudaArray_wrapper(){}
public:
	cudaArray *_array3D;

	CudaArray_wrapper(int width, int height, int depth);

	template<typename T>  
		void array3DCopy(unsigned char *img,  enum cudaMemcpyKind kind);			
};

template<typename T> void CudaArray_wrapper::array3DCopy(unsigned char *img,  enum cudaMemcpyKind kind)	// It must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice
{
	if(kind == cudaMemcpyHostToDevice)
	{
		// Allocate depthmaps array.
		cudaChannelFormatDesc fmt = cudaCreateChannelDesc<T>();	
		struct cudaExtent extent = make_cudaExtent(_width, _height, _depth);	
		CUDA_SAFE_CALL(cudaMalloc3DArray(&_array3D,&fmt,extent, cudaArrayLayered));	

		//struct cudaExtent extent = make_cudaExtent(_width, _height, _depth); //If a CUDA array is participating in the copy, the extent is defined in terms of that array's elements	
		//copy data
		struct cudaMemcpy3DParms params = {0};
		params.extent = extent;
		params.kind = /*cudaMemcpyHostToDevice*/ kind; 
		params.dstArray = _array3D;
		params.srcPtr = make_cudaPitchedPtr((void*)img,_width*sizeof(T),_width,_height);
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
		params.dstPtr = make_cudaPitchedPtr((void*)img,_width*sizeof(T),_width,_height);
		params.srcArray = _array3D;
		CUDA_SAFE_CALL(cudaMemcpy3D(&params));
	}
}

#endif
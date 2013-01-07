#ifndef ARRAY2D_WRAPPER
#define ARRAY2D_WRAPPER
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "utility_CUDA.h"

__global__ void generate_kernel_float( curandState *state, int statePitch,  float * result, int resultPitch, int width, int height, float rangeStart, float rangeEnd );
__global__ void generate_kernel_float_withDepth( curandState *state, int statePitch,  float * result, int resultPitch, int width, int height, int depth, float rangeStart, float rangeEnd );

template<class T>
class Array2D_wrapper{
public:
	T *_array2D;
	size_t _pitchData;
	Array2D_wrapper(int width, int height, int blockDim_x, int blockDim_y, int depth = 1):
			_width(width), _height(height), _blockDim_x(blockDim_x), _blockDim_y(blockDim_y),
				_depth(depth), _array2D(NULL)
	{
		CUDA_SAFE_CALL(cudaMallocPitch((void**)&_array2D, &_pitchData, _width * sizeof(T), _height * _depth));
		// compute grid and block size
		computeCUDAConfig();
	}

	~Array2D_wrapper()
	{
		if(_array2D != NULL)
			CUDA_SAFE_CALL(cudaFree((void *) _array2D));
	}
	void randNumGen(float rangeStart, float rangeEnd, curandState * devStates, int pitchState);
	size_t getWidth();
	size_t getHeight();
	size_t getDepth();


protected:
	void computeCUDAConfig();

	size_t _width;
	size_t _height;
	size_t _depth;

	dim3 _blockSize;
	int _blockDim_x;
	int _blockDim_y;
	dim3 _gridSize;

};

template<class T>
size_t Array2D_wrapper<T> :: getHeight()
{
	return _height;
}

template<class T>
size_t Array2D_wrapper<T> :: getWidth()
{
	return _width;
}

template<class T>
size_t Array2D_wrapper<T> :: getDepth()
{
	return _depth;
}

template<class T>
void Array2D_wrapper<T> :: randNumGen(float rangeStart, float rangeEnd, curandState * devStates, int pitchState)
{		
	if(_depth == 1)
		generate_kernel_float <<<_gridSize, _blockSize >>>(devStates, pitchState, _array2D, _pitchData, _width, _height, rangeStart, rangeEnd );	
	else
		generate_kernel_float_withDepth<<<_gridSize, _blockSize >>>(devStates, pitchState, _array2D, _pitchData, _width, _height, _depth, rangeStart, rangeEnd );	
}

template<class T>
void Array2D_wrapper<T>::computeCUDAConfig()
{
	_blockSize.x = _blockDim_x;
	_blockSize.y = _blockDim_y;
	_blockSize.z = 1;

	_gridSize.x = (_width - 1)/ static_cast<int>(_blockDim_x) + 1 ;
	_gridSize.y = (_height - 1)/ static_cast<int>(_blockDim_y) + 1 ;
	_gridSize.z = 1;
}

#endif
#ifndef ARRAY2D_WRAPPER
#define ARRAY2D_WRAPPER
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "utility_CUDA.h"

class Array2D_wrapper{
public:
	float *_array2D;
	Array2D_wrapper(int width, int height, int blockDim_x, int blockDim_y):
			_width(width), _height(height), _blockDim_x(blockDim_x), _blockDim_y(blockDim_y)
	{
		CUDA_SAFE_CALL(  cudaMallocPitch((void**)&_array2D, &_pitchData, _width * sizeof(float), _height));
		// compute grid and block size
		computeCUDAConfig();
		++_referenceCount;
	}

	~Array2D_wrapper()
	{
		--_referenceCount;

		if(_array2D != NULL)
			CUDA_SAFE_CALL(cudaFree((void *) _array2D));
		if(_devStates != NULL && _referenceCount == 0)  // _devState is shared among all the instances. Only delete it if no one reference it.
		{
			CUDA_SAFE_CALL(cudaFree((void *) _devStates));
			_devStates = NULL;	// set it to NULL after deletion
		}
	}
	//assign random value
	void assignRandomValue();	

	void computeCUDAConfig();
	void randNumGen(float rangeStart, float rangeEnd);	

	void randStateGen();

private:
	size_t _width;
	size_t _height;
	size_t _pitchData;
	size_t _pitchState;

	dim3 _blockSize;
	int _blockDim_x;
	int _blockDim_y;
	dim3 _gridSize;

	static curandState * _devStates;
	static int _referenceCount;

};






#endif
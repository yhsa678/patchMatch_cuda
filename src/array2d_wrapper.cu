#include "array2d_wrapper.h"

curandState *Array2D_wrapper::_devStates = NULL;
int Array2D_wrapper::_referenceCount = 0;

__global__ void setup_kernel ( curandState * state,  int width, int height, int pitch )
{
	int x = blockIdx.x * gridDim.x + threadIdx.x;
	int y = blockIdx.y * gridDim.y + threadIdx.y;

	int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	int id = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x; // unique id

/* Each thread gets same seed , a different sequence
	number , no offset */
	if( x < width && y < height)
	{
		curandState *address = (curandState *)((char*)state + y * pitch) + x;
		curand_init(id, 0, 0, address);
	}
}

__global__ void generate_kernel_float( curandState *state, int statePitch,  float * result, int resultPitch, int width, int height, float rangeStart, float rangeEnd )
{
	int x = blockIdx.x * gridDim.x + threadIdx.x;
	int y = blockIdx.y * gridDim.y + threadIdx.y;

	//int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	if( x < width && y < height)
	{
		curandState *localStateAddr = (curandState *)((char*)state + y * statePitch) + x;
		float *dataAddr = (float*)((char*)result + y * resultPitch) + x;

		curandState localState = *localStateAddr;
		/* Generate pseudo - random unsigned ints */		
		*dataAddr = (1.0f - curand_uniform(&localState)) * (rangeEnd - rangeStart) + rangeStart; // result include 0, but not 1		
		/* Copy state back to global memory */
		*localStateAddr = localState ;
	}
}

void Array2D_wrapper::randStateGen()
{
	/* Allocate space for prng states on device */
	if( _devStates == NULL)
		CUDA_SAFE_CALL(  cudaMallocPitch( ( void **)& _devStates ,  &_pitchState,  _width * sizeof( curandState ), _height));
	/* Setup prng states */
	setup_kernel <<<_gridSize, _blockSize>>>( _devStates,  _width,  _height, _pitchState );
}

void Array2D_wrapper::randNumGen(float rangeStart, float rangeEnd)
{		
	generate_kernel_float <<<_gridSize, _blockSize >>>( _devStates, _pitchState,   _array2D, _pitchData, _width, _height, rangeStart, rangeEnd );	
}

void Array2D_wrapper::computeCUDAConfig()
{
	_blockSize.x = _blockDim_x;
	_blockSize.y = _blockDim_y;
	_blockSize.z = 1;

	_gridSize.x = (_width - 1)/ static_cast<int>(_blockDim_x) + 1 ;
	_gridSize.y = (_height - 1)/ static_cast<int>(_blockDim_y) + 1 ;
	_gridSize.z = 1;
}
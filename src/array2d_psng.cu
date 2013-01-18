#include "array2d_psng.h"

__global__ void setup_kernel ( curandState * state,  int width, int height, int pitch )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	int id = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x; // unique id

/* Each thread gets same seed, a different sequence	number, no offset */
	if( x < width && y < height)
	{
		curandState *address = (curandState *)((char*)state + y * pitch) + x;
		curand_init(id, 0, 0, address);
	}
}

void Array2D_psng::randStateGen()
{
	/* Allocate space for prng states on device */
	if( _array2D == NULL)
	{
		size_t pitchData;
		CUDA_SAFE_CALL(  cudaMallocPitch( ( void **)& _array2D ,  &pitchData,  _width * sizeof(curandState), _height));
		_pitchData = static_cast<int>(pitchData);
	}

	/* Setup prng states */
	setup_kernel <<<_gridSize, _blockSize>>>( _array2D,  _width,  _height, _pitchData );
}

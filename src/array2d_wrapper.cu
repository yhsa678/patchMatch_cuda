#include "array2d_wrapper.h"

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
		*localStateAddr = localState;
	}
}

__global__ void generate_kernel_float_withDepth( curandState *state, int statePitch,  float * result, int resultPitch, int width, int height, int depth, float rangeStart, float rangeEnd )
{
	int x = blockIdx.x * gridDim.x + threadIdx.x;
	int y = blockIdx.y * gridDim.y + threadIdx.y;

	//int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	if( x < width && y < height)
	{
		curandState *localStateAddr = (curandState *)((char*)state + y * statePitch) + x;
		curandState localState = *localStateAddr;
		float *dataAddr;

		for(int i = 0; i < depth; i++)
		{
			dataAddr = (float*)((char*)result + (y * depth + i) * resultPitch) + x;
			/* Generate pseudo - random float. The rand state is also changed */		
			*dataAddr = (1.0f - curand_uniform(&localState)) * (rangeEnd - rangeStart) + rangeStart; // result include 0, but not 1		
		}

		/* Copy state back to global memory */
		*localStateAddr = localState;
	}
}


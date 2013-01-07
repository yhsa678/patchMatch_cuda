#ifndef ARRAY2D_PSNG
#define ARRAY2D_PSNG

#include "array2d_wrapper.h"

__global__ void setup_kernel( curandState * state,  int width, int height, int pitch );

class Array2D_psng : public Array2D_wrapper<curandState> // this class is not a template. Only the base class is a template
{
	// initialize initial state
	void randStateGen();

public:
	Array2D_psng(int width, int height, int blockDim_x, int blockDim_y): Array2D_wrapper(width, height, blockDim_x, blockDim_y)
	{
		randStateGen();
	}

};


#endif
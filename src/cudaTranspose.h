#include <stdio.h>
#include <cuda_runtime.h>

class cudaTranspose{
public:
	static void transpose2dData(float * input, float *output, int width, int height, int pitchInput, int pitchOutput);

};



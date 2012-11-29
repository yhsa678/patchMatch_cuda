#ifndef PATCHMATCH_H 
#define PATCHMATCH_H 

#include <cuda_runtime.h>
#include "cudaArray_wrapper.h"
#include "array2d_wrapper.h"
#include <vector>
#include <iostream>
#include "image.h"

class PatchMatch
{
public:
	float _nearRange;
	float _farRange;
	int _halfWindowSize;
	
	int _blockDim_x;
	int _blockDim_y;
	
	int _numOfImages;
	CudaArray_wrapper *_allImages_cudaArrayWrapper;
	Array2D_wrapper * _depthMap;
	Array2D_wrapper * _SPMap; // selection probability map

	unsigned char *_imageDataBlock;

//--------------------------------------------------
	PatchMatch(const std::vector<Image> &allImage, float nearRange, float farRange, int halfWindowSize, int blockDim_x, int blockDim_y);
	void run();

	~PatchMatch();
	
	void testTextureArray();

	

};



	 














#endif
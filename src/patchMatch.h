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
	int _refImageId;
	//int _numOfImages;
	CudaArray_wrapper *_allImages_cudaArrayWrapper;
	CudaArray_wrapper *_refImages_cudaArrayWrapper;

	Array2D_wrapper * _depthMap;
	Array2D_wrapper * _SPMap; // selection probability map
	
	
	int _numOfTargetImages;
	int _maxWidth;
	int _maxHeight;
	unsigned char *_imageDataBlock;

	int _refWidth;
	int _refHeight;
	unsigned char *_refImageDataBlock;

	float *_transformHH;

//--------------------------------------------------
	PatchMatch( std::vector<Image> &allImage, float nearRange, float farRange, int halfWindowSize, int blockDim_x, int blockDim_y, int refImageId);
	void run();

	~PatchMatch();
	
	void testTextureArray();

private:
	void PatchMatch::copyData(const std::vector<Image> &allImage, int referenceId);

};



	 














#endif
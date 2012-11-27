#ifndef PATCHMATCH_H 
#define PATCHMATCH_H 

#include <cuda_runtime.h>
#include "cudaArray_wrapper.h"
#include <vector>
#include <iostream>
#include "image.h"

class PatchMatch
{
public:

	CudaArray_wrapper *_allImages_cudaArrayWrapper;
	char *_imageDataBlock;

	PatchMatch(const std::vector<Image> &allImage);
	

	~PatchMatch();
	


};

	 














#endif
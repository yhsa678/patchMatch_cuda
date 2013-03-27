#ifndef PATCHMATCH_H 
#define PATCHMATCH_H 

#include <cuda_runtime.h>
#include "cudaArray_wrapper.h"
#include "array2d_wrapper.h"
#include "array2d_psng.h"
#include "array2d_refImg.h"
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

	dim3 _blockSize;
	dim3 _gridSize;

	//int _numOfImages;
	CudaArray_wrapper *_allImages_cudaArrayWrapper;
	CudaArray_wrapper *_refImages_cudaArrayWrapper;

	Array2D_wrapper<float> * _depthMap;
	Array2D_wrapper<float> * _depthMapT;
	Array2D_wrapper<float> * _SPMap; // selection probability map
	Array2D_wrapper<float> * _SPMapT;

	Array2D_wrapper<float> * _matchCost;
	Array2D_wrapper<float> * _matchCostT;

	Array2D_wrapper<float> * _stateProb;
	Array2D_wrapper<float> * _stateProbT;

	Array2d_refImg *_refImage;
	Array2d_refImg *_refImageT;

	CudaArray_wrapper *_transformArray;

	//Array2D_wrapper<float> * _randDepth;
	//Array2D_wrapper<float> * _randDepthT;

	Array2D_psng *_psngState;
	//Array2D_psng *_psngStateT;
	
	int _numOfTargetImages;
	int _maxWidth;
	int _maxHeight;
	unsigned char *_imageDataBlock;

	int _refWidth;
	int _refHeight;
	unsigned char *_refImageDataBlock;


	float *_transformHH;
	int _numOfSamples;

	float _SPMAlpha;
	float _gaussianSigma;
	int _numOfIterations;
//--------------------------------------------------
	PatchMatch( std::vector<Image> &allImage, float nearRange, float farRange, int halfWindowSize, int blockDim_x, int blockDim_y, int refImageId, int numOfSamples, float SPMAlpha, float gaussianSigma, int numOfIterations);

	void runPatchMatch();

	~PatchMatch();
	

private:
	void copyData(const std::vector<Image> &allImage, int referenceId);
	void transpose(Array2D_wrapper<float> *input, Array2D_wrapper<float> *output);
	void transposeForward();
	void transposeBackward();

	void computeCUDAConfig(int width, int height, int blockDim_x, int blockDim_y);
	template<int WINDOWSIZES> void run();
};



	 














#endif
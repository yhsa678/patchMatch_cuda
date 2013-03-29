#include "patchMatch.h"
#include "cudaTranspose.h"
#include "utility_CUDA.h"
#include "GaussianBlurCUDA.h"
#include  <sstream> 

#define MAX_WINDOW_SIZE	53 

#define FIX_STATE_PROB (0.999f)
#define CHANGE_STATE_PROB (1.0f - FIX_STATE_PROB)

#define HANDLE_BOUNDARY

template<int WINDOWSIZES>
__global__ void topToDown( float *, float *, float *, float *, int, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated, unsigned int, float);


template<int WINDOWSIZES>
__global__ void downToTop(float *, float *, float *, float *, int, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated, unsigned int, float, uchar *usedImgsID, unsigned usedImgsIDPitchData);

template<int WINDOWSIZES>
__global__ void computeAllCostGivenDepth(float *matchCost, int SPMapPitch, float *refImg, float *refImgI, float *refImgII, int refImgPitch, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch,
	unsigned int _numOfTargetImages);

texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> allImgsTexture;
texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> refImgTexture;
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> transformTexture; 

//__constant__ float transformHH[MAX_NUM_IMAGES * 9 * 2];
__constant__ float orientation[3];
__constant__ float inverseK[4];

#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))
#define SET_BIT(var,pos)( (var) |= (1 << (pos) ))
#define N 32 

void PatchMatch::computeCUDAConfig(int width, int height, int blockDim_x, int blockDim_y)
{
	_blockSize.x = blockDim_x;
	_blockSize.y = blockDim_y;
	_blockSize.z = 1;

	_gridSize.x = (width - 1)/ static_cast<int>(blockDim_x) + 1 ;
	//_gridSize.y = (height - 1)/ static_cast<int>(blockDim_y) + 1 ;
	_gridSize.y = 1;
	_gridSize.z = 1;
}

void PatchMatch::copyData(const std::vector<Image> &allImage, int referenceId)
{
	//input the data and extract consective data block
	int numOfChannels =  allImage[0]._imageData.channels();
	_maxWidth = 0;
	_maxHeight = 0;
	for(unsigned int i = 0; i < allImage.size(); i++)
	{
		if(i != referenceId)
		{
			if(allImage[i]._imageData.cols > _maxWidth)
			{
				_maxWidth = allImage[i]._imageData.cols;
			}
			if(allImage[i]._imageData.rows > _maxHeight)
			{
				_maxHeight = allImage[i]._imageData.rows;
			}
		}	
	}
	// ---------- assign memory, copy data
	size_t sizeOfBlock = static_cast<size_t>(_maxWidth) * static_cast<size_t>(numOfChannels)
		* static_cast<size_t>(_maxHeight)  * static_cast<size_t>(_numOfTargetImages);
	_imageDataBlock = new unsigned char[sizeOfBlock]();
	// copy row by row
	//char *dest = _imageDataBlock;
	int dataBlockId = 0;
	for(unsigned int i = 0; i < allImage.size(); i++)
	{	
		if(i != referenceId)
		{
			unsigned char *dest = _imageDataBlock + (_maxWidth * numOfChannels * _maxHeight) * dataBlockId ;
			unsigned char *source = allImage[i]._imageData.data;
			for( int j = 0; j < _maxHeight; j++)
			{
				if(j < allImage[i]._imageData.rows)
				{
					memcpy( (void *)dest, (void *)source,  allImage[i]._imageData.cols * numOfChannels * sizeof(unsigned char));	
					dest += (_maxWidth * numOfChannels);
					source += allImage[i]._imageData.step;
				}	
			}
			++dataBlockId;
		}
	}
	
	// for the reference image
	_refWidth = allImage[referenceId]._imageData.cols;
	_refHeight = allImage[referenceId]._imageData.rows;
	if(numOfChannels != allImage[referenceId]._imageData.channels())
	{ std::cout<< "reference Image has different number of channels with target images"<< std::endl; 
	 exit(EXIT_FAILURE);}
	_refImageDataBlock = new unsigned char[_refWidth * _refHeight * numOfChannels](); 
	if( allImage[referenceId]._imageData.isContinuous())
		memcpy((void *)_refImageDataBlock, allImage[referenceId]._imageData.data, _refWidth * _refHeight * numOfChannels * sizeof(unsigned char));
	else
	{
		unsigned char *dest = _refImageDataBlock;
		unsigned char *source = allImage[referenceId]._imageData.data;
		for(int i = 0; i < _refHeight; i++)
		{
			memcpy((void*)dest, (void*)source,  _refWidth * numOfChannels * sizeof(unsigned char) );	
			source += allImage[i]._imageData.step;
			dest += (_refWidth * numOfChannels);
		}
	}

	_transformHH = new float[18 * _numOfTargetImages]();
	int offset = 0;
	for(unsigned int i = 0; i< allImage.size(); i++)
	{
		if(i != referenceId)
		{
			memcpy((void*)(_transformHH+offset), (void*)allImage[i]._H1.data, 9 * sizeof(float));
			offset += 9;
			memcpy((void*)(_transformHH+offset), (void*)allImage[i]._H2.data, 9 * sizeof(float));
			offset += 9;
		}
	}
} 

PatchMatch::PatchMatch( std::vector<Image> &allImage, float nearRange, float farRange, int halfWindowSize, int blockDim_x, int blockDim_y, int refImageId, int numOfSamples, float SPMAlpha, float gaussianSigma, int numOfIterations, float orientationX, float orientationZ): 
	_imageDataBlock(NULL), _allImages_cudaArrayWrapper(NULL), _nearRange(nearRange), _farRange(farRange), _halfWindowSize(halfWindowSize), _blockDim_x(blockDim_x), _blockDim_y(blockDim_y), _refImageId(refImageId),
		_depthMap(NULL), _SPMap(NULL), _psngState(NULL), _depthMapT(NULL), _SPMapT(NULL), _numOfSamples(numOfSamples), _refImage(NULL), _refImageT(NULL), _SPMAlpha(SPMAlpha), _gaussianSigma(gaussianSigma),
		_numOfIterations(numOfIterations), _orientationX(orientationX), _orientationZ(orientationZ)
{
	_numOfTargetImages = static_cast<int>(allImage.size()) - 1;
	if(_numOfTargetImages == 0)
	{
		std::cout<< "Error: at least 2 images are needed for stereo" << std::endl;
		exit(EXIT_FAILURE);
	}
	// using reference image id to update H1 and H2 for each image
	for(unsigned int i = 0; i < allImage.size(); i++)
		allImage[i].init_relative( allImage[refImageId], _orientationX, _orientationZ );

	// find maximum size of each dimension
	copyData(allImage, _refImageId);
	
	// upload H matrix
	//cudaMemcpyToSymbol("transformHH", _transformHH , sizeof(float) * 18 * _numOfTargetImages, 0, cudaMemcpyHostToDevice);
	float orientationDataBlock[3] = {_orientationX, 0, _orientationZ};
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(orientation, orientationDataBlock , sizeof(float) * 3, 0, cudaMemcpyHostToDevice));
	float inverseKDataBlock[4]; 
	inverseKDataBlock[0] = allImage[_refImageId]._inverseK.at<float>(0,0);  inverseKDataBlock[1] = allImage[_refImageId]._inverseK.at<float>(0,2);
	inverseKDataBlock[2] = allImage[_refImageId]._inverseK.at<float>(1,1);  inverseKDataBlock[3] = allImage[_refImageId]._inverseK.at<float>(1,2);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(inverseK,	inverseKDataBlock, sizeof(float) * 4, 0, cudaMemcpyHostToDevice));


	// initialize depthmap and SP(selection probability) map
	_depthMap = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y);
	_matchCost = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y, _numOfTargetImages);

	_psngState = new Array2D_psng(_refWidth, _refHeight, _blockDim_x, _blockDim_y);

	_depthMap->randNumGen(_nearRange, _farRange, _psngState->_array2D, _psngState->_pitchData);
	_depthMapT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, _blockDim_y);
//	_matchCostT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, blockDim_y, _numOfTargetImages);

	_usedImgsID = new Array2D_wrapper<uchar>(_refWidth, _refHeight, _blockDim_x, _blockDim_y, _numOfSamples);


	// reference image
	_refImage = new Array2d_refImg(_refWidth, _refHeight, blockDim_x, blockDim_y, _refImageDataBlock);
	_refImage->filterImage(halfWindowSize);
	_refImageT = new Array2d_refImg(_refHeight, _refWidth, blockDim_x, blockDim_y);
	transpose(_refImage->_refImage_sum_I, _refImageT->_refImage_sum_I);
	transpose(_refImage->_refImage_sum_II, _refImageT->_refImage_sum_II);
	transpose(_refImage->_refImageData, _refImageT->_refImageData);
	
	// ---------- initialize array
	_allImages_cudaArrayWrapper = new CudaArray_wrapper(_maxWidth, _maxHeight, _numOfTargetImages);
	_refImages_cudaArrayWrapper = new CudaArray_wrapper(_refWidth, _refHeight, 1);
	_transformArray = new CudaArray_wrapper(18, _numOfTargetImages, 1);

	// ---------- upload image data to GPU
	_allImages_cudaArrayWrapper->array3DCopy<unsigned char>(_imageDataBlock, cudaMemcpyHostToDevice);
	_refImages_cudaArrayWrapper->array3DCopy<unsigned char>(_refImageDataBlock, cudaMemcpyHostToDevice);
	_transformArray->array3DCopy_float(_transformHH, cudaMemcpyHostToDevice, 0);		// the last param is not used

	// attach to texture so that the kernel can access the data
	allImgsTexture.addressMode[0] = cudaAddressModeBorder; allImgsTexture.addressMode[1] = cudaAddressModeBorder; 
	allImgsTexture.addressMode[2] = cudaAddressModeBorder;
	allImgsTexture.filterMode = cudaFilterModeLinear;	allImgsTexture.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(allImgsTexture, _allImages_cudaArrayWrapper->_array3D));	// bind to texture	
	
	refImgTexture.addressMode[0] = cudaAddressModeBorder; refImgTexture.addressMode[1] = cudaAddressModeBorder; 
	refImgTexture.addressMode[2] = cudaAddressModeBorder;
	refImgTexture.filterMode = cudaFilterModeLinear;	refImgTexture.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(refImgTexture, _refImages_cudaArrayWrapper->_array3D));	// bind to

	transformTexture.addressMode[0] = cudaAddressModeBorder; transformTexture.addressMode[1] = cudaAddressModeBorder;
	transformTexture.addressMode[2] = cudaAddressModeBorder;
	transformTexture.filterMode = cudaFilterModePoint;  transformTexture.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(transformTexture, _transformArray->_array3D));
}

PatchMatch::~PatchMatch()
{
	if(_imageDataBlock != NULL)
		delete []_imageDataBlock;
	if(_refImageDataBlock != NULL)
		delete []_refImageDataBlock;
	if(_transformHH != NULL)
		delete []_transformHH;
	if(_allImages_cudaArrayWrapper != NULL)
		delete _allImages_cudaArrayWrapper;
	if(_SPMap != NULL)
		delete _SPMap;
	if(_depthMap != NULL)
		delete _depthMap;
	if(_psngState != NULL)
		delete _psngState;
	if(_SPMapT != NULL)
		delete _SPMapT;
	if(_depthMapT != NULL)
		delete _depthMapT;
	if(_refImage != NULL)
		delete _refImage;
	if(_refImageT != NULL)
		delete _refImageT;
	if(_transformArray != NULL)
		delete _transformArray;
}

void PatchMatch::transpose(Array2D_wrapper<float> *input, Array2D_wrapper<float> *output)
{
	for(int d = 0; d < input->getDepth(); d++)
	{
		cudaTranspose::transpose2dData( input->_array2D + d * input->_pitchData/sizeof(float) * input->getHeight(),			// **** here sizeof(float) may be able to be changed to corresponds to the datatype of input->array2d
										output->_array2D + d * output->_pitchData/sizeof(float) * output->getHeight(),
										input->getWidth(), input->getHeight(), input->_pitchData, output->_pitchData);
		CudaCheckError();
	}
}

void PatchMatch::transposeForward()
{
	transpose(_depthMap, _depthMapT);
	_matchCostT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, _blockDim_y, _numOfTargetImages);
	transpose(_matchCost, _matchCostT);
	delete _matchCost; _matchCost = NULL;
}

void PatchMatch::transposeBackward()
{
	transpose(_depthMapT, _depthMap);
	_matchCost = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y, _numOfTargetImages);
	transpose(_matchCostT, _matchCost);
	delete _matchCostT; _matchCostT = NULL;
}

void PatchMatch::runPatchMatch()
{
	if(_halfWindowSize > (MAX_WINDOW_SIZE-1)/2 || _halfWindowSize < 0)
	{
		std::cout<< "half of the window size cannot be larger than 26 or smaller than 0" << std::endl;
		std::cout<< "stereo is not done" << std::endl;
		return;
	}

	int windowSize = _halfWindowSize * 2 + 1;

	switch(windowSize)
	{
		case(1):	run<1 >(); break;
		case(3):	run<3 >(); break;
		case(5):	run<5 >(); break;	
		case(7):	run<7 >(); break;
		case(9):	run<9 >(); break;
		case(11):	run<11>(); break;
		case(13):	run<13>(); break;
		case(15):	run<15>(); break;
		case(17):	run<17>(); break;
		case(19):	run<19>(); break;
		case(21):	run<21>(); break;
		case(23):	run<23>(); break;
		case(25):	run<25>(); break;
		case(27):	run<27>(); break;
		case(29):	run<29>(); break;
		case(31):	run<31>(); break;
		case(33):	run<33>(); break;
		case(35):	run<35>(); break;
		case(37):	run<37>(); break;
		case(39):	run<39>(); break;
		case(41):	run<41>(); break;
		case(43):	run<43>(); break;
		case(45):	run<45>(); break;
		case(47):	run<47>(); break;
		case(49):	run<49>(); break;
		case(51):	run<51>(); break;
		case(53):	run<53>(); break;
	}
}


template<int WINDOWSIZES> void PatchMatch::run()
{
	int numOfSamples;
	bool isRotated;
	std::cout<< "started" << std::endl;
	std::cout<< "the window size is: " << WINDOWSIZES << std::endl;
	std::cout<< "number of iterations is: " << _numOfIterations << std::endl;
	CudaTimer t;

	float SPMAlphaSquare = _SPMAlpha * _SPMAlpha;
	int sizeOfdynamicSharedMemory = sizeof(float) * N * _numOfTargetImages  + sizeof(unsigned int) * (_numOfTargetImages/32 + 1) * N;
	// calculate amount of shared memory used in total:
	int totalNumOfSharedMemUsed = ((3 * N) + (N) + (N) + (N * 3 * WINDOWSIZES) + N * sizeof(curandState))*sizeof(float) + sizeOfdynamicSharedMemory;
	std::cout<< "totolNumber of shared memory used: " << totalNumOfSharedMemUsed << " bytes" << std::endl;
	std::cout<< "totalNumber of dynamic shared: " << sizeOfdynamicSharedMemory << std::endl;
	checkSharedMem(totalNumOfSharedMemUsed);
	for(int i = 0; i < _numOfIterations; i++)
	{
	// left to right sweep
//-----------------------------------------------------------
		std::cout<< "Iteration " << i << " starts" << std::endl;
		if(i == 0)
			numOfSamples = 1; // ****
		else
			numOfSamples = _numOfSamples;
		
		t.startRecord();
		computeCUDAConfig(_depthMapT->getWidth(), _depthMapT->getHeight(), N, 1);
		transposeForward();

		_SPMapT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, _blockDim_y, _numOfTargetImages);
		if(i == 0)
		{
			computeAllCostGivenDepth<WINDOWSIZES><<<_gridSize, _blockSize>>>(_matchCostT->_array2D, _matchCostT->_pitchData ,_refImageT->_refImageData->_array2D, _refImageT->_refImage_sum_I->_array2D, _refImageT->_refImage_sum_II->_array2D,
			_refImageT->_refImage_sum_I->_pitchData, _depthMapT->getWidth(), _depthMapT->getHeight(), _depthMapT->_array2D, _depthMapT->_pitchData, _numOfTargetImages);
			CudaCheckError();
		}		
		isRotated = true;
		topToDown<WINDOWSIZES><<<_gridSize, _blockSize, sizeOfdynamicSharedMemory>>>(_matchCostT->_array2D, _refImageT->_refImageData->_array2D,  _refImageT->_refImage_sum_I->_array2D, _refImageT->_refImage_sum_II->_array2D, _refImageT->_refImage_sum_I->_pitchData,
			_depthMapT->getWidth(), _depthMapT->getHeight(), _depthMapT->_array2D, _depthMapT->_pitchData, 
			_SPMapT->_array2D, _SPMapT->_pitchData, 
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated, _numOfTargetImages, SPMAlphaSquare);
		delete _SPMapT; _SPMapT = NULL;
////----------------------------------------------------------------------------------------------
//	// top to bottom sweep 
		transposeBackward();
		_SPMap = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y, _numOfTargetImages);
		computeCUDAConfig(_depthMap->getWidth(), _depthMap->getHeight(), N, 1);
		isRotated = false;
		topToDown<WINDOWSIZES><<<_gridSize, _blockSize, sizeOfdynamicSharedMemory>>>(_matchCost->_array2D, _refImage->_refImageData->_array2D, _refImage->_refImage_sum_I->_array2D, _refImage->_refImage_sum_II->_array2D, _refImage->_refImage_sum_I->_pitchData,
			_depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_array2D, _depthMap->_pitchData, 
			_SPMap->_array2D, _SPMap->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated, _numOfTargetImages, SPMAlphaSquare);
		CudaCheckError();
		delete _SPMap; _SPMap = NULL;

	//////////// right to left sweep
		transposeForward();
		computeCUDAConfig(_depthMapT->getWidth(), _depthMapT->getHeight(), N, 1);
		isRotated = true;
		_SPMapT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, _blockDim_y, _numOfTargetImages);
		downToTop<WINDOWSIZES><<<_gridSize, _blockSize, sizeOfdynamicSharedMemory>>>(_matchCostT->_array2D, _refImageT->_refImageData->_array2D, _refImageT->_refImage_sum_I->_array2D, _refImageT->_refImage_sum_II->_array2D, _refImageT->_refImage_sum_I->_pitchData,
			_depthMapT->getWidth(), _depthMapT->getHeight(), _depthMapT->_array2D, _depthMapT->_pitchData, 
			_SPMapT->_array2D, _SPMapT->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated, _numOfTargetImages, SPMAlphaSquare, NULL, 0);
		delete _SPMapT; _SPMapT = NULL;
		CudaCheckError();
		
	//////// bottom to top sweep
		transposeBackward();
		_SPMap = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y, _numOfTargetImages);
		computeCUDAConfig(_depthMap->getWidth(), _depthMap->getHeight(), N, 1);
		isRotated = false;

		if(i == _numOfIterations - 1)
			downToTop<WINDOWSIZES><<<_gridSize, _blockSize, sizeOfdynamicSharedMemory>>>(_matchCost->_array2D, _refImage->_refImageData->_array2D, _refImage->_refImage_sum_I->_array2D, _refImage->_refImage_sum_II->_array2D, _refImage->_refImage_sum_I->_pitchData,
			_depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_array2D, _depthMap->_pitchData, 
			_SPMap->_array2D, _SPMap->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated, _numOfTargetImages, SPMAlphaSquare, _usedImgsID->_array2D, _usedImgsID->_pitchData );
		else
			downToTop<WINDOWSIZES><<<_gridSize, _blockSize, sizeOfdynamicSharedMemory>>>(_matchCost->_array2D, _refImage->_refImageData->_array2D, _refImage->_refImage_sum_I->_array2D, _refImage->_refImage_sum_II->_array2D, _refImage->_refImage_sum_I->_pitchData,
			_depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_array2D, _depthMap->_pitchData, 
			_SPMap->_array2D, _SPMap->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated, _numOfTargetImages, SPMAlphaSquare, NULL, 0);
		delete _SPMap; _SPMap = NULL;
		t.stopRecord();

	}

	// do refinement:


	
	/*for(int i = 0; i< _numOfTargetImages; i++)
	{
		std::stringstream ss; ss<<i;
		std::string fileName = "_SPMap"+ ss.str() + ".txt";
		_SPMap->saveToFile(fileName, i);
	}*/
	std::cout<< "ended " << std::endl;
}

template<typename T>
//inline __device__ float accessPitchMemory(float *data, int pitch, int row, int col)
inline __device__ T accessPitchMemory(T *data, int pitch, int row, int col)
{
	return *((T*)((char*)data + pitch*row) + col);
}

template<typename T>
//inline __device__ void writePitchMemory(float *data, int pitch, int row, int col, float value )
inline __device__ void writePitchMemory(T *data, int pitch, int row, int col, T value )
{
	*((T*)((char*)data + pitch*row) + col) = value;
}

inline __device__ float drawRandNum(curandState *state, int statePitch, int col, int row, float rangeNear, float rangeFar)
{
	curandState *localStateAddr = (curandState *)((char*)state + row * statePitch) + col;	
	curandState localState = *localStateAddr;
	float randNum = curand_uniform(&localState) * (rangeFar - rangeNear) + rangeNear;
	*localStateAddr = localState;
	return randNum;
}


template<int WINDOWSIZES> 
inline __device__ float computeNCC(const int &threadId, const float *refImg_I, const float *refImg_sum_I, const float *refImg_sum_II, 
	//const int &imageId, const float &centerRow, const float &centerCol, const float &depth, const int &halfWindowSize, const bool &isRotated, const float& refImgWidth, const float& refImgHeight)
	const int &imageId, const float &rowMinusHalfwindowPlusHalf, const float &colMinusHalfwindowPlusHalf, const float &depth, const bool &isRotated, const int &halfWindowSize ,
	const int &refImageWidth, const int &refImageHeight, const float &scale)
	// here the return resutls are 1-NCC, so the range is [0, 2], the smaller value, the better color consistency
{
	float sum_I_Iprime = 0;
	float sum_Iprime_Iprime = 0;
	float sum_Iprime = 0;
	float Iprime;
	
	//float transform[9]; 
	//float *transformBase = transformHH + 18 * imageId;
	int numOfPixels = 0;
	//for(int i = 0; i<9; i++)
	//{
	//	transform[0] = (transformBase)[0] - (transformBase)[9]/depth;
	//	transform[1] = (transformBase)[1] - (transformBase)[10]/depth;
	//	transform[2] = (transformBase)[2] - (transformBase)[11]/depth;
	//	transform[3] = (transformBase)[3] - (transformBase)[12]/depth;
	//	transform[4] = (transformBase)[4] - (transformBase)[13]/depth;
	//	transform[5] = (transformBase)[5] - (transformBase)[14]/depth;
	//	transform[6] = (transformBase)[6] - (transformBase)[15]/depth;
	//	transform[7] = (transformBase)[7] - (transformBase)[16]/depth;
	//	transform[8] = (transformBase)[8] - (transformBase)[17]/depth;
	//	//transform[i] = (transformHH + 18 * imageId)[i] - (transformHH + 18 * imageId)[i+9]/depth;
	//}
	
	float imageId_y = static_cast<float>(imageId) + 0.5f;
	float transform[9];
	{
		transform[0] = tex2DLayered( transformTexture, 0.5, imageId_y, 0) - tex2DLayered( transformTexture, 9.5,  imageId_y, 0)/depth/scale;
		transform[1] = tex2DLayered( transformTexture, 1.5, imageId_y, 0) - tex2DLayered( transformTexture, 10.5, imageId_y, 0)/depth/scale;
		transform[2] = tex2DLayered( transformTexture, 2.5, imageId_y, 0) - tex2DLayered( transformTexture, 11.5, imageId_y, 0)/depth/scale;
		transform[3] = tex2DLayered( transformTexture, 3.5, imageId_y, 0) - tex2DLayered( transformTexture, 12.5, imageId_y, 0)/depth/scale;
		transform[4] = tex2DLayered( transformTexture, 4.5, imageId_y, 0) - tex2DLayered( transformTexture, 13.5, imageId_y, 0)/depth/scale;
		transform[5] = tex2DLayered( transformTexture, 5.5, imageId_y, 0) - tex2DLayered( transformTexture, 14.5, imageId_y, 0)/depth/scale;
		transform[6] = tex2DLayered( transformTexture, 6.5, imageId_y, 0) - tex2DLayered( transformTexture, 15.5, imageId_y, 0)/depth/scale;
		transform[7] = tex2DLayered( transformTexture, 7.5, imageId_y, 0) - tex2DLayered( transformTexture, 16.5, imageId_y, 0)/depth/scale;
		transform[8] = tex2DLayered( transformTexture, 8.5, imageId_y, 0) - tex2DLayered( transformTexture, 17.5, imageId_y, 0)/depth/scale;
	}
	/*for(int i = 0; i < 8; i++)
	{
		if(transform1[i] != transform[0])
			printf("It is different: %f,  %f\n", transform1[i], transform[i]);
	}*/


	float z;
	float col_prime;
	float row_prime;
	float base_col_prime;
	float base_row_prime;
	float base_z;
	if(!isRotated)
	{
		/*z = base_z =         transform[6] * (centerCol - halfWindowSize + 0.5) + transform[7] * (centerRow - halfWindowSize + 0.5) + transform[8];
		col_prime = base_col_prime = transform[0] * (centerCol - halfWindowSize + 0.5) + transform[1] * (centerRow - halfWindowSize + 0.5) + transform[2];
		row_prime = base_row_prime = transform[3] * (centerCol - halfWindowSize + 0.5) + transform[4] * (centerRow - halfWindowSize + 0.5) + transform[5];*/
		z = base_z =         transform[6] * colMinusHalfwindowPlusHalf + transform[7] * rowMinusHalfwindowPlusHalf + transform[8];
		col_prime = base_col_prime = transform[0] * colMinusHalfwindowPlusHalf + transform[1] * rowMinusHalfwindowPlusHalf + transform[2];
		row_prime = base_row_prime = transform[3] * colMinusHalfwindowPlusHalf + transform[4] * rowMinusHalfwindowPlusHalf + transform[5];
	}
	else
	{
		/*z = base_z =         transform[6] * (centerRow - halfWindowSize + 0.5) + transform[7] * (centerCol - halfWindowSize + 0.5) + transform[8];
		col_prime = base_col_prime = transform[0] * (centerRow - halfWindowSize + 0.5) + transform[1] * (centerCol - halfWindowSize + 0.5) + transform[2];
		row_prime = base_row_prime = transform[3] * (centerRow - halfWindowSize + 0.5) + transform[4] * (centerCol - halfWindowSize + 0.5) + transform[5];*/
		z = base_z =         transform[6] * rowMinusHalfwindowPlusHalf + transform[7] * colMinusHalfwindowPlusHalf + transform[8];
		col_prime = base_col_prime = transform[0] * rowMinusHalfwindowPlusHalf + transform[1] * colMinusHalfwindowPlusHalf + transform[2];
		row_prime = base_row_prime = transform[3] * rowMinusHalfwindowPlusHalf + transform[4] * colMinusHalfwindowPlusHalf + transform[5];
	}

	//int localSharedMemRow = 0;
	//int localSharedMemCol ;
	//for(float row = centerRow - halfWindowSize; row <= centerRow + halfWindowSize; row++) // y
	int refImg_I_Ind;
	int base_refImg_I_Ind = (refImg_I_Ind = N - halfWindowSize + threadId);

	float sum_Iprime_row;
	float sum_Iprime_Iprime_row;
	float sum_I_Iprime_row; 

#ifdef HANDLE_BOUNDARY
	float localRowMinusHalfwindowPlusHalf = rowMinusHalfwindowPlusHalf;
	float localColMinusHalfwindowPlusHalf;// = colMinusHalfwindowPlusHalf;
#endif


	for(int localSharedMemRow = 0; localSharedMemRow < WINDOWSIZES; localSharedMemRow ++)
	{
		sum_Iprime_row = 0.0f;
		sum_Iprime_Iprime_row = 0.0f;
		sum_I_Iprime_row = 0.0f;
		
#ifdef HANDLE_BOUNDARY
		if( localRowMinusHalfwindowPlusHalf >= 0.5 && localRowMinusHalfwindowPlusHalf <= refImageHeight- 0.5)
#endif
		{
#ifdef HANDLE_BOUNDARY
			localColMinusHalfwindowPlusHalf = colMinusHalfwindowPlusHalf;
#endif
			for(int localSharedMemCol = 0; localSharedMemCol < WINDOWSIZES; localSharedMemCol++)
			{
				// do transform to get the new position
#ifdef HANDLE_BOUNDARY
				if(localColMinusHalfwindowPlusHalf >=0.5 && localColMinusHalfwindowPlusHalf <= refImageWidth - 0.5)
#endif
				{
					Iprime = tex2DLayered(allImgsTexture, col_prime/z + 0.5f, row_prime/z + 0.5f, imageId); // textures are not rotated
					sum_Iprime_Iprime_row += (Iprime * Iprime);
					sum_Iprime_row += Iprime;
					sum_I_Iprime_row += (Iprime * refImg_I[refImg_I_Ind++]);
					numOfPixels++;
				}
#ifdef HANDLE_BOUNDARY
				++localColMinusHalfwindowPlusHalf;
#endif
				if(!isRotated)
				{
					z += transform[6];
					col_prime += transform[0];
					row_prime += transform[3];
				}
				else
				{
					z += transform[7];
					col_prime += transform[1];
					row_prime += transform[4];
				}
			}
			sum_Iprime += sum_Iprime_row;
			sum_Iprime_Iprime += sum_Iprime_Iprime_row;
			sum_I_Iprime += sum_I_Iprime_row;
		}
#ifdef HANDLE_BOUNDARY
		++localRowMinusHalfwindowPlusHalf;
#endif

		refImg_I_Ind = ( base_refImg_I_Ind += (3 * N));
		if(!isRotated) 
		{
			z = (base_z += transform[7] );
			col_prime = (base_col_prime += transform[1]); 
			row_prime = (base_row_prime += transform[4]);
		}
		else
		{
			z = (base_z += transform[6] );
			col_prime = (base_col_prime += transform[0] ); 
			row_prime = (base_row_prime += transform[3] );
		}
	}	
	//float numOfPixels = static_cast<float>((halfWindowSize * 2 + 1)*(halfWindowSize * 2 + 1));
	//float numOfPixels = static_cast<float>(windowSize * windowSize );

//	float cost = (refImg_sum_II[threadId] - refImg_sum_I[threadId] * refImg_sum_I[threadId]/ numOfPixels) 
	//	* (sum_Iprime_Iprime - sum_Iprime * sum_Iprime/ numOfPixels); 
	//float cost = ((refImg_sum_II[threadId]*numOfPixels - refImg_sum_I[threadId] * refImg_sum_I[threadId]) 
	//	* (sum_Iprime_Iprime*numOfPixels - sum_Iprime * sum_Iprime)); 
	if(sum_Iprime_Iprime == 0.0f)
		return 2.0f;

	float cost1 = refImg_sum_II[threadId] - refImg_sum_I[threadId] * refImg_sum_I[threadId]/ (float)numOfPixels;
	float cost2 = sum_Iprime_Iprime - sum_Iprime * sum_Iprime/ (float)numOfPixels; 
	cost1 = cost1 < 0.00001? 0.0f : cost1;
	cost2 = cost2 < 0.00001? 0.0f : cost2;
	float cost = sqrt(cost1 * cost2);
	if(cost == 0)
		return 1.0f; // very small color consistency
	else
	{
		//float norminator = sum_I_Iprime * numOfPixels - refImg_sum_I[threadId] * sum_Iprime;
		//return 1 -  abs(norminator)/(cost);
		cost = 1 - (sum_I_Iprime -  refImg_sum_I[threadId] * sum_Iprime/ (float)numOfPixels )/(cost);
		return cost;
	}
}

inline __device__ int findMinCost(float *cost)
{
	int idx = cost[0] < cost[1]? 0:1;
	idx = cost[idx] < cost[2]? idx:2;
	return idx;
}

template<int WINDOWSIZES>
inline __device__ void readImageIntoSharedMemory(float *refImg_I, int row, int col, const int& threadId, const bool &isRotated) 
	// here assumes N is bigger than HALFBLOCK
{
	// the size of the data block: 3N * (2 * halfblock + 1)
	if(!isRotated)
	{
		//row -= halfWindowSize;
		//row -= HALFBLOCK;
		row -= (WINDOWSIZES-1)/2;
		col -= N;
		for(int i = 0; i < WINDOWSIZES; i++)
		{

			refImg_I[threadId + i * 3 * N] = tex2DLayered( refImgTexture, col + 0.5f, row + 0.5f, 0);
			col += N; 
			refImg_I[threadId + i * 3 * N + N] = tex2DLayered( refImgTexture, col + 0.5f, row + 0.5f, 0);
			col += N;
			refImg_I[threadId + i * 3 * N + 2 * N] = tex2DLayered( refImgTexture, col + 0.5f, row + 0.5f , 0);
			col -= 2 * N; // go back to the 1st col
			++row; // increase one row
		}
#if N>32
		__syncthreads();
#endif
	}
	else
	{
		//row -= HALFBLOCK;
		//row -= halfWindowSize;
		row -= (WINDOWSIZES-1)/2; 
		col -= N;
		for(int i = 0; i < WINDOWSIZES; i++)
		{

			refImg_I[threadId + i * 3 * N] = tex2DLayered( refImgTexture, row + 0.5f, col + 0.5f, 0);
			col += N; 
			refImg_I[threadId + i * 3 * N + N] = tex2DLayered( refImgTexture, row + 0.5f, col + 0.5f, 0);
			col += N;
			refImg_I[threadId + i * 3 * N + 2 * N] = tex2DLayered( refImgTexture, row + 0.5f, col + 0.5f , 0);
			col -= 2 * N; // go back to the 1st col
			++row; // increase one row
		}
#if N>32
		__syncthreads();
#endif
	}
}

__device__ void computeMessageBackward(float *normalizedSPMap, const int &row, const int &col, const int &threadId, float *matchCost, 
	const int &SPMapPitch, const float &SPMAlphaSquare, const int &refImageHeight, const int &_numOfTargetImages, curandState *localState)
{
	float emission;
	const float x = 0.83f;
	float uniformProb = exp(-0.5f * x * x/SPMAlphaSquare);

	for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
	{
		//float unseenProb = 1.0f - normalizedSPMap[imageId * N + threadId];
		emission = accessPitchMemory(matchCost, SPMapPitch, imageId * refImageHeight + row, col);
		if(emission == 2.0f)
			//emission = 1.2;
			emission = curand_uniform(&localState[threadId]) * 2.0f;
		emission = exp( -0.5 * emission * emission/SPMAlphaSquare);

		float Zn0 = normalizedSPMap[imageId * N + threadId] * emission * FIX_STATE_PROB + (1.0f - normalizedSPMap[imageId * N + threadId]) * uniformProb * CHANGE_STATE_PROB; // probability of Zn = 0 (no occlusion)
		float Zn1 = normalizedSPMap[imageId * N + threadId] * emission * CHANGE_STATE_PROB + (1.0f - normalizedSPMap[imageId * N + threadId]) * uniformProb * FIX_STATE_PROB; 
		normalizedSPMap[imageId * N + threadId] = Zn0/(Zn0 + Zn1);
	}

}

__device__ void computeMessageForward(float *normalizedSPMap, const int &row, const int &col, const int &threadId, float *matchCost, 
	const int &SPMapPitch, const float &SPMAlphaSquare, const int &refImageHeight, const int &_numOfTargetImages, curandState *localState )
{
	//float sum = 0.0f;
	float emission;
	const float x = 0.83f;
	float uniformProb = exp(-0.5f * x * x/SPMAlphaSquare);

	for(int imageId = 0; imageId < _numOfTargetImages; imageId ++)
	{
		// update beta_hat: 1) read in cost		2) update
		//float unseenProb = 1.0f - normalizedSPMap[imageId * N + threadId];  

		emission = accessPitchMemory(matchCost, SPMapPitch, imageId * refImageHeight + row, col);
		if(emission == 2.0f)
			emission = curand_uniform(&localState[threadId]) * 2.0f;
		emission = exp( -0.5 * emission * emission/SPMAlphaSquare);

		float Zn0 = (normalizedSPMap[imageId * N + threadId] * FIX_STATE_PROB + (1-normalizedSPMap[imageId * N + threadId]) * CHANGE_STATE_PROB) * emission;
		float Zn1 = (normalizedSPMap[imageId * N + threadId] * CHANGE_STATE_PROB + (1-normalizedSPMap[imageId * N + threadId]) * FIX_STATE_PROB)* uniformProb;

		normalizedSPMap[imageId * N + threadId] = Zn0/(Zn0 + Zn1);
	}
}

template<int WINDOWSIZES>
__global__ void depthRefinement(float *matchCost, int SPMapPitch, float *refImg, float *refImgI, float *refImgII, int refImgPitch, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch,
	unsigned int _numOfTargetImages, uchar *usedImgsId, int usedImgsIdPitchData,  int numOfSamples, curandState *randState, int randStatePitch )
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadId = threadIdx.x;

	__shared__ float refImg_sum_I[N];
	__shared__ float refImg_sum_II[N];
	__shared__ float refImg_I[N * 3 * WINDOWSIZES];
	__shared__ float depth_current_array[N];
	__shared__ float depth_new[N];
	__shared__ curandState localState[N];
	localState[threadId] = *(randState + col);

	float rowMinusHalfwindowPlusHalf = 0.0f - (WINDOWSIZES-1)/2 + 0.5f;
	float colMinusHalfwindowPlusHalf = (float)col - (WINDOWSIZES-1)/2 + 0.5f;

	for(int row = 0; row < refImageHeight; ++row)
	{
		readImageIntoSharedMemory<WINDOWSIZES>(refImg_I, row, col, threadId, true);
		if(col < refImageWidth)
		{
			refImg_sum_I[threadId] = accessPitchMemory(refImgI, refImgPitch, row, col);
			refImg_sum_II[threadId] = accessPitchMemory(refImgII, refImgPitch, row, col);			
			depth_current_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col);
			float oldCost = accessPitchMemory(matchCost, SPMapPitch, row,col);
			float newCost;
			float bestDepth;

			for(int i = 0; i < 10; i++)
			{
				newCost = 0;
				// randomly change the depth:
				depth_new[threadId] = depth_current_array[threadId] + curand_normal(&localState[threadId]) * 0.05f ;
				depth_new[threadId] = depth_new[threadId] <= 0? depth_current_array[threadId] : depth_new[threadId];

				uchar imageId;
				for(int j = 0; j < numOfSamples; j++)
				{
					imageId = accessPitchMemory(usedImgsId, usedImgsIdPitchData, j * refImageHeight + row, col);
					newCost += computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II, static_cast<int>(imageId), rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_new[threadId], true, (WINDOWSIZES-1)/2, refImageWidth, refImageHeight);

				}
				newCost /= numOfSamples;

				if(newCost < oldCost)
				{
					bestDepth = depth_new[threadId];
					oldCost = newCost;
				}
			}
			writePitchMemory(depthMap, depthMapPitch, row, col, bestDepth);
			++rowMinusHalfwindowPlusHalf;
		}
	}	*(randState + col) = localState[threadId];
}

template<int WINDOWSIZES>
__global__ void computeAllCostGivenDepth(float *matchCost, int SPMapPitch, float *refImg, float *refImgI, float *refImgII, int refImgPitch, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch,
	unsigned int _numOfTargetImages)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadId = threadIdx.x;

	__shared__ float refImg_sum_I[N];
	__shared__ float refImg_sum_II[N];
	__shared__ float refImg_I[N *3 * WINDOWSIZES];
	__shared__ float depth_current_array[N];
	//float rowMinusHalfwindowPlusHalf = 0.0f - halfWindowSize + 0.5f;
	float rowMinusHalfwindowPlusHalf = 0.0f - (WINDOWSIZES-1)/2 + 0.5f;
	float colMinusHalfwindowPlusHalf = (float)col - (WINDOWSIZES-1)/2 + 0.5f;

	float scale;
	for(int row = 0; row < refImageHeight; ++row)
	{

		readImageIntoSharedMemory<WINDOWSIZES>( refImg_I, row, col, threadId, true);
		if(col < refImageWidth)
		{
			
			scale = orientation[0] * (inverseK[0] * (row + 0.5) + inverseK[1] ) 
					+ orientation[1] * (inverseK[2] * (col + 0.5) + inverseK[3])
					+ orientation[2];

			refImg_sum_I[threadId] = accessPitchMemory(refImgI, refImgPitch, row, col);
			refImg_sum_II[threadId] = accessPitchMemory(refImgII, refImgPitch, row, col);
			depth_current_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col);

			float cost1stRow;
			for(int imageId = 0; imageId < _numOfTargetImages; imageId ++)
			{
				// compute cost for current depth
				cost1stRow = computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_current_array[threadId], true, (WINDOWSIZES-1)/2, refImageWidth, refImageHeight, scale);
				writePitchMemory(matchCost, SPMapPitch,  row + imageId * refImageHeight, col, cost1stRow);
			}
			++rowMinusHalfwindowPlusHalf;
		}
	}
}

template<int WINDOWSIZES>
__global__ void topToDown(float *matchCost, float *refImg, float *refImgI, float *refImgII, int refImgPitch, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated, unsigned int _numOfTargetImages, float SPMAlphaSquare)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int threadId = threadIdx.x;
	extern __shared__ float normalizedSPMap[];
	
	float colMinusHalfwindowPlusHalf = (float)col - halfWindowSize + 0.5f;
	__shared__ float depth_array[3 * N];
	float * depth_former_array = depth_array;
	float * depth_current_array = depth_array + N;
	float * randDepth = depth_array + 2*N;

	__shared__ float refImg_sum_I[N];
	__shared__ float refImg_sum_II[N];
	__shared__ float refImg_I[N *3 * WINDOWSIZES];

	unsigned int s = (_numOfTargetImages >> 5) + 1; // 5 is because each int type has 32 bits, and divided by 32 is equavalent to shift 5. s is number of bytes used to save selected images

	unsigned int* selectedImages = (unsigned int*)(normalizedSPMap +  N * _numOfTargetImages);
	//float* forwardMessageTemp = normalizedSPMap + N * _numOfTargetImages;	// 10 is number of Target Images

	__shared__ curandState localState[N];		
	if(col < refImageWidth)
	{
		if(!isRotated)
			localState[threadId] = *(randState + col);
		else
			localState[threadId] = *((curandState*)((char*)randState + col * randStatePitch));
		depth_former_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, 0, col); 	// depth for 1st element
	}
	//--------------------------------------------------------------------------------------------
	float rowMinusHalfwindowPlusHalf = 0.0f - halfWindowSize + 0.5f;
	//----------------------------------------------------------------------------------------------
	// precompute the message from down to top. SPMap is used to save the message! Then these backward messages are going to be used later
	//float beta_hat = 1.0f;
	float scale;
	if(col < refImageWidth)
	{
		for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			normalizedSPMap[imageId * N + threadId ] = 0.5f; 
		for(int row = refImageHeight - 1; row >=0; row--)
		{
			for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			{
				writePitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col, normalizedSPMap[imageId * N + threadId]); // the backward message is saved here
			}				
			computeMessageBackward(normalizedSPMap, row, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);
		}
		for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			writePitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + 0, col, normalizedSPMap[imageId * N + threadId]); // the backward message is saved here

		// next compute the forward message
		for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			normalizedSPMap[imageId * N + threadId ] = 0.5f; 
		computeMessageForward(normalizedSPMap, 0, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);

		if(!isRotated)
			scale = orientation[0] * (inverseK[0] * (col + 0.5) + inverseK[1] ) 
				  + orientation[1] * (inverseK[2] * (0.5) + inverseK[3])
				  + orientation[2];
		else
			scale = orientation[0] * (inverseK[0] * 0.5 + inverseK[1])
				  + orientation[1] * (inverseK[2] * (col + 0.5) + inverseK[3])
				  + orientation[2];

	}
	//----------------------------------------------------------------------------------------------
	
	for( int row = 1; row < refImageHeight; ++row)
	{
		readImageIntoSharedMemory<WINDOWSIZES>( refImg_I, row, col, threadId, isRotated);

		if(col < refImageWidth)
		{
			if(!isRotated)
				scale += orientation[1] * inverseK[2];
			else
				scale += orientation[0] * inverseK[0];

			//for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			//	forwardMessageTemp[imageId * N + threadId] = normalizedSPMap[imageId * N + threadId ]; 
			for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
				writePitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row - 1, col, normalizedSPMap[imageId * N + threadId]); // the backward message is saved here

			//computeMessageForward(forwardMessageTemp, row, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);
			computeMessageForward(normalizedSPMap, row, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);


			// read in the backward message and compute the current state.
			for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			{
				//float zn0 = forwardMessageTemp[imageId * N + threadId] * accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col);
				//float zn1 = (1.0f - forwardMessageTemp[imageId * N + threadId]) * (1.0f- accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col));
				float zn0 = normalizedSPMap[imageId * N + threadId] * accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col);
				float zn1 = (1.0f - normalizedSPMap[imageId * N + threadId]) * (1.0f- accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col));
				normalizedSPMap[imageId * N + threadId] = zn0/(zn0+zn1);
			}
			for(int i = 1; i<_numOfTargetImages; i++)		
				//forwardMessageTemp[i * N + threadId] += forwardMessageTemp[(i-1) * N + threadId ];
				normalizedSPMap[i * N + threadId] += normalizedSPMap[(i-1) * N + threadId ];
			// normalize
			for(int i = 0; i<_numOfTargetImages; i++)
				//forwardMessageTemp[i * N + threadId] /= forwardMessageTemp[N * (_numOfTargetImages -1) + threadId];			
				normalizedSPMap[i * N + threadId] /= normalizedSPMap[N * (_numOfTargetImages -1) + threadId];			

			++rowMinusHalfwindowPlusHalf;
			refImg_sum_I[threadId] = accessPitchMemory(refImgI, refImgPitch, row, col);
			refImg_sum_II[threadId] = accessPitchMemory(refImgII, refImgPitch, row, col);

			depth_current_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col); 
			for(int i = 0; i<s; i++)
				selectedImages[threadId + i * N] = 0;	// initialized to false
			//---------------------------------
			// draw samples and set the bit to 0
			float numOfTestedSamples = 0;
			float cost[3] = {0.0f};
			// here it is better to generate a random depthmap
			randDepth[threadId] = curand_uniform(&localState[threadId]) * (depthRangeFar - depthRangeNear) + depthRangeNear;

			unsigned int pos;
			for(int j = 0; j < numOfSamples; j++)
			{
				float randNum = curand_uniform(&localState[threadId]);

				int imageId = -1;				
				for(unsigned int i = 0; i < _numOfTargetImages; i++)
				{
					if(randNum <= normalizedSPMap[i * N + threadId ])
					//if(randNum <= forwardMessageTemp[i * N + threadId ])
					{
						unsigned int &stateByte = selectedImages[(i>>5) * N + threadId];
						pos = i - (sizeof(unsigned int) << 3) * (i>>5);  //(i - i /32 * numberOfBitsPerInt)
						//if(!CHECK_BIT(stateByte,pos))
						{
							imageId = i;
							numOfTestedSamples++;
						}
						// then set the bit
						SET_BIT(stateByte, pos);	// 
						break;
					}
				}
				// image id is i( id != -1). Test the id using NCC, with 3 different depth. 				
				//if(imageId != -1)
				{
					cost[0] +=  computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_former_array[threadId], isRotated, halfWindowSize, refImageWidth, refImageHeight, scale);			// accumulate the cost
					cost[1] += accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
					cost[2] +=  computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, randDepth[threadId], isRotated, halfWindowSize, refImageWidth, refImageHeight, scale);	
				}
			}	
			// find the minimum cost id, and then put cost into global memory 
			//cost[0] /= numOfTestedSamples; cost[1] /= numOfTestedSamples; cost[2] /= numOfTestedSamples;
			numOfTestedSamples = 1.0f/numOfTestedSamples;
			cost[0] *= numOfTestedSamples; cost[1] *= numOfTestedSamples; cost[2] *= numOfTestedSamples;

			int idx = findMinCost(cost);
			float bestDepth = depth_array[threadId + N * idx];
			writePitchMemory(depthMap, depthMapPitch, row, col, bestDepth);
			// swap depth former and depth current
			depth_former_array[threadId] = bestDepth;
			// Here I need to calculate SPMap based on the best depth, and put it into SPMap
			//float variance_inv = 1.0/(0.2 * 0.2);
			// compute the cost for bestDepth
			for(int imageId = 0; imageId < _numOfTargetImages; imageId ++)
			{
				if(idx != 1)
				{	
					cost[0] = computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, isRotated, halfWindowSize, refImageWidth, refImageHeight, scale);
					writePitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col, cost[0]);
				} 
				else
				{
					cost[0] = accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
				}
			} 

			for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
				normalizedSPMap[imageId * N + threadId] = accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row - 1, col );

			computeMessageForward(normalizedSPMap, row, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);
		}
	}
	if(col < refImageWidth)
	{
		if(!isRotated)
			*(randState + col) = localState[threadId];
		else
			*((curandState*)((char*)randState + col * randStatePitch)) = localState[threadId];
	}
	
}

template<int WINDOWSIZES>
__global__ void downToTop(float *matchCost, float *refImg, float *refImgI, float *refImgII, int refImgPitch, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated, unsigned int _numOfTargetImages, float SPMAlphaSquare,
	uchar *usedImgsID, int usedImgsIDPitchData )
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int threadId = threadIdx.x;
	extern __shared__ float normalizedSPMap[];

	float colMinusHalfwindowPlusHalf = (float)col - halfWindowSize + 0.5f;
	__shared__ float depth_array[3 * N];
	float * depth_former_array = depth_array;
	float * depth_current_array = depth_array + N;
	float * randDepth = depth_array + 2*N;

	__shared__ float refImg_sum_I[N];
	__shared__ float refImg_sum_II[N];
	__shared__ float refImg_I[N *3 * WINDOWSIZES];	// WINDOWSIZE SHOULD BE SMALLER LESS THAN N

	unsigned int s = (_numOfTargetImages >> 5) + 1; // 5 is because each int type has 32 bits, and divided by 32 is equavalent to shift 5. s is number of bytes used to save selected images
	unsigned int* selectedImages = (unsigned int*)(normalizedSPMap +  N * _numOfTargetImages);
	//float* forwardMessageTemp = normalizedSPMap + N * _numOfTargetImages;	// 10 is number of Target Images

	__shared__ curandState localState[N];
	if(col < refImageWidth)
	{
		if(!isRotated)
			localState[threadId] = *(randState + col);
		else
			localState[threadId] = *((curandState*)((char*)randState + col * randStatePitch));
		depth_former_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, refImageHeight - 1, col); 	// depth for 1st element
	}

	float rowMinusHalfwindowPlusHalf = refImageHeight - 1.0f - halfWindowSize + 0.5f;
	//--------------------------------------------------------------------------
	float scale;
	if(col < refImageWidth)
	{
		for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			normalizedSPMap[imageId * N + threadId ] = 0.5f; 
		for(int row = 0; row <= refImageHeight - 1; row++)
		{
			for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			{
				writePitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col, normalizedSPMap[imageId * N + threadId]); // the backward message is saved here
			}
			computeMessageBackward(normalizedSPMap, row, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);
		}
		for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			writePitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + refImageHeight - 1, col, normalizedSPMap[imageId * N + threadId]); // the backward message is saved here

		// next compute forward message:
		for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			normalizedSPMap[imageId * N + threadId ] = 0.5f; 

		computeMessageForward(normalizedSPMap, refImageHeight -1, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);

		if(!isRotated)
			scale = orientation[0] * (inverseK[0] * (col + 0.5) + inverseK[1] ) 
				  + orientation[1] * (inverseK[2] * (refImageHeight - 0.5) + inverseK[3])
				  + orientation[2];
		else
			scale = orientation[0] * (inverseK[0] * (refImageHeight - 0.5f) + inverseK[1])
			      + orientation[1] * (inverseK[2] * (col + 0.5) + inverseK[3])
				  + orientation[2];
	}


	//--------------------------------------------------------------------------
	//__shared__ float forwardMessageTemp[N * 10];
	for(int row = refImageHeight - 2; row >=0; --row)
	{
		readImageIntoSharedMemory<WINDOWSIZES>( refImg_I, row, col, threadId, isRotated);

		if(col < refImageWidth)
		{
			if(!isRotated)
				scale -= inverseK[2] * orientation[1];
			else
				scale -= inverseK[0] * orientation[0];

			//for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			//	forwardMessageTemp[imageId * N + threadId] = normalizedSPMap[imageId * N + threadId ]; 
			for( int imageId = 0; imageId < _numOfTargetImages; imageId++)
				writePitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row + 1, col, normalizedSPMap[imageId * N + threadId]);

			//computeMessageForward(forwardMessageTemp, row, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);
			computeMessageForward(normalizedSPMap, row, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);

			// read in the backward message and compute the current state.
			for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
			{
				//float zn0 = forwardMessageTemp[imageId * N + threadId] * accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col);
				//float zn1 = (1.0f - forwardMessageTemp[imageId * N + threadId]) * (1.0f- accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col));
				float zn0 = normalizedSPMap[imageId * N + threadId] * accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col);
				float zn1 = (1.0f - normalizedSPMap[imageId * N + threadId]) * (1.0f- accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row, col));
				//forwardMessageTemp[imageId * N + threadId] = zn0/(zn0+zn1);
				normalizedSPMap[imageId * N + threadId] = zn0/(zn0+zn1);
			}
			for(int i = 1; i<_numOfTargetImages; i++)		
				//forwardMessageTemp[i * N + threadId] += forwardMessageTemp[(i-1) * N + threadId ];
				normalizedSPMap[i * N + threadId] += normalizedSPMap[(i-1) * N + threadId ];
			// normalize
			for(int i = 0; i<_numOfTargetImages; i++)
				//forwardMessageTemp[i * N + threadId] /= forwardMessageTemp[N * (_numOfTargetImages -1) + threadId];			
				normalizedSPMap[i * N + threadId] /= normalizedSPMap[N * (_numOfTargetImages -1) + threadId];			


			--rowMinusHalfwindowPlusHalf;
			refImg_sum_I[threadId] = accessPitchMemory(refImgI, refImgPitch, row, col);
			refImg_sum_II[threadId] = accessPitchMemory(refImgII, refImgPitch, row, col);

			depth_current_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col); 
			for(int i = 0; i<s; i++)
				selectedImages[threadId + i * N] = 0;	// initialized to false
			//---------------------------------

			// draw samples and set the bit to 0
			float numOfTestedSamples = 0.0f;
			float cost[3] = {0.0f};

			// here it is better to generate a random depthmap
			randDepth[threadId] = curand_uniform(&localState[threadId]) * (depthRangeFar - depthRangeNear) + depthRangeNear;
			for(int j = 0; j < numOfSamples; j++)
			{
				float randNum = curand_uniform(&localState[threadId]); 

				int imageId = -1;				
				for(int i = 0; i < _numOfTargetImages; i++)
				{
					if(randNum <= normalizedSPMap[i * N + threadId ])
					//if(randNum <= forwardMessageTemp[i * N + threadId ])
					{
						unsigned int &stateByte = selectedImages[(i>>5) * N + threadId];
						unsigned int pos = i - (sizeof(int) << 3) * (i>>5);  //(i - i /32 * numberOfBitsPerInt)
						//if(!CHECK_BIT(stateByte,pos))
						{
							imageId = i;
							numOfTestedSamples++;
						}
						// then set the bit
						SET_BIT(stateByte, pos);	// 
						break;
					}
				}
				// image id is i( id != -1). Test the id using NCC, with 3 different depth. 				
				//if(imageId != -1)
				if(usedImgsID != NULL)
				{
					//printf("enter\n");
					writePitchMemory( usedImgsID, usedImgsIDPitchData, row + refImageHeight * j, col, static_cast<uchar>(imageId));
				}
				{
					cost[0] +=  computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_former_array[threadId], isRotated, halfWindowSize, refImageWidth, refImageHeight, scale);			// accumulate the cost
					cost[1] += accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
					cost[2] +=  computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, randDepth[threadId], isRotated, halfWindowSize, refImageWidth, refImageHeight,scale);
				}
			}	
			// find the minimum cost id, and then put cost into global memory 
			numOfTestedSamples = 1.0f/numOfTestedSamples;
			cost[0] *= numOfTestedSamples; cost[1] *= numOfTestedSamples; cost[2] *= numOfTestedSamples;
		
			int idx = findMinCost(cost);
			float bestDepth = depth_array[threadId + N * idx];
			writePitchMemory(depthMap, depthMapPitch, row, col, bestDepth);
			depth_former_array[threadId] = bestDepth;

			for(int imageId = 0; imageId < _numOfTargetImages; imageId ++)
			{
				if(idx != 1)
				{
					cost[0] = computeNCC<WINDOWSIZES> (threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, isRotated, halfWindowSize, refImageWidth, refImageHeight, scale);
					writePitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col, cost[0]);
				}
				else
				{
					cost[0] = accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
				}
				//cost[0] = exp(-0.5 * cost[0] * cost[0] / (SPMAlphaSquare));
				//writePitchMemory(SPMap, SPMapPitch,row + imageId * refImageHeight, col, cost[0]); // write SPMap
			} // for

			for(int imageId = 0; imageId < _numOfTargetImages; imageId++)
				normalizedSPMap[imageId *N + threadId] = accessPitchMemory(SPMap, SPMapPitch, imageId * refImageHeight + row + 1, col);

			computeMessageForward(normalizedSPMap, row, col, threadId, matchCost, SPMapPitch, SPMAlphaSquare, refImageHeight, _numOfTargetImages, localState);

		} // if(col < refImageWidth)
	} // for
	
	if(col < refImageWidth)
	{
		if(!isRotated)
			*(randState + col) = localState[threadId];
		else
			*((curandState*)((char*)randState + col * randStatePitch)) = localState[threadId];
	}
}





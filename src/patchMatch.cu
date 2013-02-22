#include "patchMatch.h"
#include "cudaTranspose.h"
#include "utility_CUDA.h"
#include "GaussianBlurCUDA.h"
#include  <sstream> 

#define MAX_NUM_IMAGES 128
#define MAX_WINDOW_SIZE	53 
//#define HANDLE_BOUNDARY

template<int WINDOWSIZES>
__global__ void topToDown(bool isFirstStart, float *, float *, float *, float *, int, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated, unsigned int, float);


template<int WINDOWSIZES>
__global__ void downToTop(bool isFirstStart, float *, float *, float *, float *, int, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated, unsigned int, float);

texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> allImgsTexture;
texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> refImgTexture;
__constant__ float transformHH[MAX_NUM_IMAGES * 9 * 2];

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
	int sizeOfBlock = _maxWidth * numOfChannels * _maxHeight  * _numOfTargetImages;
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
	_refImageDataBlock = new unsigned char[_refWidth * _refHeight * numOfChannels]; 
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

	std::cout<< static_cast<int>(_refImageDataBlock[_refWidth * 577 + 2]) << std::endl;

	_transformHH = new float[18 * _numOfTargetImages];
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

PatchMatch::PatchMatch( std::vector<Image> &allImage, float nearRange, float farRange, int halfWindowSize, int blockDim_x, int blockDim_y, int refImageId, int numOfSamples, float SPMAlpha, float gaussianSigma): 
	_imageDataBlock(NULL), _allImages_cudaArrayWrapper(NULL), _nearRange(nearRange), _farRange(farRange), _halfWindowSize(halfWindowSize), _blockDim_x(blockDim_x), _blockDim_y(blockDim_y), _refImageId(refImageId),
		_depthMap(NULL), _SPMap(NULL), _psngState(NULL), _depthMapT(NULL), _SPMapT(NULL), _numOfSamples(numOfSamples), _refImage(NULL), _refImageT(NULL), _SPMAlpha(SPMAlpha), _gaussianSigma(gaussianSigma)
{
	_numOfTargetImages = static_cast<int>(allImage.size()) - 1;
	if(_numOfTargetImages == 0)
	{
		std::cout<< "Error: at least 2 images are needed for stereo" << std::endl;
		exit(EXIT_FAILURE);
	}
	// using reference image id to update H1 and H2 for each image
	for(unsigned int i = 0; i < allImage.size(); i++)
		allImage[i].init_relative( allImage[refImageId] );

	// find maximum size of each dimension
	copyData(allImage, _refImageId);
	
	// upload H matrix
	cudaMemcpyToSymbol("transformHH", _transformHH , sizeof(float) * 18 * _numOfTargetImages, 0, cudaMemcpyHostToDevice);

	// initialize depthmap and SP(selection probability) map
	_depthMap = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y);
	_SPMap = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y, _numOfTargetImages);
	_matchCost = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y, _numOfTargetImages);
	

	_psngState = new Array2D_psng(_refWidth, _refHeight, _blockDim_x, _blockDim_y);

	_depthMap->randNumGen(_nearRange, _farRange, _psngState->_array2D, _psngState->_pitchData);
	_SPMap->randNumGen(0.0f, 1.0f, _psngState->_array2D, _psngState->_pitchData);
	//viewData1DDevicePointer( _SPMap->_array2D, 100);

	_depthMapT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, _blockDim_y);
	_SPMapT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, _blockDim_y, _numOfTargetImages);
	_matchCostT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, blockDim_y, _numOfTargetImages);

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

	// ---------- upload image data to GPU
	_allImages_cudaArrayWrapper->array3DCopy<unsigned char>(_imageDataBlock, cudaMemcpyHostToDevice);
	_refImages_cudaArrayWrapper->array3DCopy<unsigned char>(_refImageDataBlock, cudaMemcpyHostToDevice);
	// attach to texture so that the kernel can access the data
	allImgsTexture.addressMode[0] = cudaAddressModeBorder; allImgsTexture.addressMode[1] = cudaAddressModeBorder; 
	allImgsTexture.addressMode[2] = cudaAddressModeBorder;
	allImgsTexture.filterMode = cudaFilterModeLinear;	allImgsTexture.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(allImgsTexture, _allImages_cudaArrayWrapper->_array3D));	// bind to texture	
	
	refImgTexture.addressMode[0] = cudaAddressModeBorder; refImgTexture.addressMode[1] = cudaAddressModeBorder; 
	refImgTexture.addressMode[2] = cudaAddressModeBorder;
	refImgTexture.filterMode = cudaFilterModeLinear;	refImgTexture.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(refImgTexture, _refImages_cudaArrayWrapper->_array3D));	// bind to

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
	transpose(_SPMap, _SPMapT);
	transpose(_matchCost, _matchCostT);
}

void PatchMatch::transposeBackward()
{
	transpose(_depthMapT, _depthMap);
	transpose(_SPMapT, _SPMap);
	transpose(_matchCostT, _matchCost);
}

void PatchMatch::runPatchMatch()
{
	if(_halfWindowSize > 26 || _halfWindowSize < 0)
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
		default: { std::cout<< "The window size is not valid\n"; }
	}
}


template<int WINDOWSIZES> void PatchMatch::run()
{
	int numOfSamples;
	bool isRotated;
	std::cout<< "started" << std::endl;
	std::cout<< "the window size is: " << WINDOWSIZES << std::endl;
	CudaTimer t;
	GaussianBlurCUDA gFilter(_refWidth, _refHeight, _gaussianSigma);
	GaussianBlurCUDA gFilterT(_depthMapT->getWidth(), _depthMapT->getHeight(), _gaussianSigma);

	float SPMAlphaSquare = _SPMAlpha * _SPMAlpha;
	bool isFirstStart = true;
	int sizeOfdynamicSharedMemory = sizeof(float) * N * _numOfTargetImages + sizeof(unsigned int) * (_numOfTargetImages/32 + 1) * N;

	for(int i = 0; i < 1; i++)
	{
	// left to right sweep
//-----------------------------------------------------------
		std::cout<< "Iteration " << i << " starts" << std::endl;
		if(i == 0)
			numOfSamples = 1; // ****
		else
			numOfSamples = _numOfSamples;
		
		t.startRecord();
		transposeForward();
		computeCUDAConfig(_depthMapT->getWidth(), _depthMapT->getHeight(), N, 1);
		isRotated = true;
		topToDown<WINDOWSIZES><<<_gridSize, _blockSize, sizeOfdynamicSharedMemory>>>(isFirstStart, _matchCostT->_array2D, _refImageT->_refImageData->_array2D,  _refImageT->_refImage_sum_I->_array2D, _refImageT->_refImage_sum_II->_array2D, _refImageT->_refImage_sum_I->_pitchData,
			_depthMapT->getWidth(), _depthMapT->getHeight(), _depthMapT->_array2D, _depthMapT->_pitchData, 
			_SPMapT->_array2D, _SPMapT->_pitchData, 
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated, _numOfTargetImages, SPMAlphaSquare);
		CudaCheckError();
		gFilterT.FilterMultipleImages( _SPMapT->_array2D, _SPMapT->_pitchData, _SPMapT->getDepth());
		
		//t.stopRecord();
////-----------------------------------------------------------
//	// top to bottom sweep 
		//t.startRecord();
		transposeBackward();
	//	isFirstStart = false;
	//	computeCUDAConfig(_depthMap->getWidth(), _depthMap->getHeight(), N, 1);
	//	isRotated = false;
	//	topToDown<WINDOWSIZES><<<_gridSize, _blockSize, sizeOfdynamicSharedMemory>>>(isFirstStart, _matchCost->_array2D, _refImage->_refImageData->_array2D, _refImage->_refImage_sum_I->_array2D, _refImage->_refImage_sum_II->_array2D, _refImage->_refImage_sum_I->_pitchData,
	//		_depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_array2D, _depthMap->_pitchData, 
	//		_SPMap->_array2D, _SPMap->_pitchData,
	//		numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated, _numOfTargetImages, SPMAlphaSquare);
	//	gFilter.FilterMultipleImages(_SPMap->_array2D, _SPMap->_pitchData, _SPMap->getDepth());

	////////////// right to left sweep
	//	transposeForward();
	//	computeCUDAConfig(_depthMapT->getWidth(), _depthMapT->getHeight(), N, 1);
	//	isRotated = true;
	//	downToTop<WINDOWSIZES><<<_gridSize, _blockSize, sizeOfdynamicSharedMemory>>>(isFirstStart, _matchCostT->_array2D, _refImageT->_refImageData->_array2D, _refImageT->_refImage_sum_I->_array2D, _refImageT->_refImage_sum_II->_array2D, _refImageT->_refImage_sum_I->_pitchData,
	//		_depthMapT->getWidth(), _depthMapT->getHeight(), _depthMapT->_array2D, _depthMapT->_pitchData, 
	//		_SPMapT->_array2D, _SPMapT->_pitchData,
	//		numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated, _numOfTargetImages, SPMAlphaSquare);
	//	CudaCheckError();
	//	gFilterT.FilterMultipleImages( _SPMapT->_array2D, _SPMapT->_pitchData, _SPMapT->getDepth());
	//	
	////////// bottom to top sweep
	//	transposeBackward();
	//	computeCUDAConfig(_depthMap->getWidth(), _depthMap->getHeight(), N, 1);
	//	isRotated = false;
	//	downToTop<WINDOWSIZES><<<_gridSize, _blockSize, sizeOfdynamicSharedMemory>>>(isFirstStart, _matchCost->_array2D, _refImage->_refImageData->_array2D, _refImage->_refImage_sum_I->_array2D, _refImage->_refImage_sum_II->_array2D, _refImage->_refImage_sum_I->_pitchData,
	//		_depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_array2D, _depthMap->_pitchData, 
	//		_SPMap->_array2D, _SPMap->_pitchData,
	//		numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated, _numOfTargetImages, SPMAlphaSquare);
	//	gFilter.FilterMultipleImages(_SPMap->_array2D, _SPMap->_pitchData, _SPMap->getDepth());
	//	t.stopRecord();
	}
	
	/*for(int i = 0; i< _numOfTargetImages; i++)
	{
		std::stringstream ss; ss<<i;
		std::string fileName = "_SPMap"+ ss.str() + ".txt";
		_SPMap->saveToFile(fileName, i);
	}*/
	std::cout<< "ended " << std::endl;
}

inline __device__ float accessPitchMemory(float *data, int pitch, int row, int col)
{
	return *((float*)((char*)data + pitch*row) + col);
}

inline __device__ void writePitchMemory(float *data, int pitch, int row, int col, float value )
{
	*((float*)((char*)data + pitch*row) + col) = value;
}

inline __device__ float drawRandNum(curandState *state, int statePitch, int col, int row, float rangeNear, float rangeFar)
{
	curandState *localStateAddr = (curandState *)((char*)state + row * statePitch) + col;	
	curandState localState = *localStateAddr;
	float randNum = curand_uniform(&localState) * (rangeFar - rangeNear) + rangeNear;
	*localStateAddr = localState;
	return randNum;
}

//inline __device__ void doTransform(float &col_prime, float &row_prime, float col, float row, int imageId, float *transform)
//{
//	//float *base = &transformHH[0] +  18 * imageId;
//	//float z = (base[6] - base[15]/depth) * col + (base[7] - base[16]/depth) * row + (base[8] - base[17]/depth);
//	//*col_prime = ((base[0] - base[9]/depth) * col + (base[1] - base[10]/depth) * row + (base[2] - base[11]/depth))/z;
//	//*row_prime = ((base[3] - base[12]/depth) * col + (base[4] - base[13]/depth) * row + (base[5] - base[14]/depth))/z;
//	float z = transform[6] * col + transform[7] * row + transform[8];
//	col_prime = (transform[0] * col + transform[1] * row + transform[2])/z;
//	row_prime = (transform[3] * col + transform[4] * row + transform[5])/z;
//}
template<int WINDOWSIZES> 
inline __device__ float computeNCCNew(const int &threadId, const float *refImg_I, const float *refImg_sum_I, const float *refImg_sum_II, 
	//const int &imageId, const float &centerRow, const float &centerCol, const float &depth, const int &halfWindowSize, const bool &isRotated, const float& refImgWidth, const float& refImgHeight)
	const int &imageId, const float &rowMinusHalfwindowPlusHalf, const float &colMinusHalfwindowPlusHalf, const float &depth, const int &windowSize, const bool &isRotated, const int &halfWindowSize ,
	const int &refImageWidth, const int &refImageHeight)
	// here the return resutls are 1-NCC, so the range is [0, 2], the smaller value, the better color consistency
{
	float sum_I_Iprime = 0;
	float sum_Iprime_Iprime = 0;
	float sum_Iprime = 0;
	float Iprime;
	
	float transform[9]; 
	float *transformBase = transformHH + 18 * imageId;
	int numOfPixels = 0;
	//for(int i = 0; i<9; i++)
	{
		transform[0] = (transformBase)[0] - (transformBase)[9]/depth;
		transform[1] = (transformBase)[1] - (transformBase)[10]/depth;
		transform[2] = (transformBase)[2] - (transformBase)[11]/depth;
		transform[3] = (transformBase)[3] - (transformBase)[12]/depth;
		transform[4] = (transformBase)[4] - (transformBase)[13]/depth;
		transform[5] = (transformBase)[5] - (transformBase)[14]/depth;
		transform[6] = (transformBase)[6] - (transformBase)[15]/depth;
		transform[7] = (transformBase)[7] - (transformBase)[16]/depth;
		transform[8] = (transformBase)[8] - (transformBase)[17]/depth;
		//transform[i] = (transformHH + 18 * imageId)[i] - (transformHH + 18 * imageId)[i+9]/depth;
	}
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

	float refImgSum_I_temp = 0.0f;
	float refImgSum_I_I_temp = 0.0f;

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
					//float xxx = refImg_I[refImg_I_Ind++];
					sum_I_Iprime_row += (Iprime * refImg_I[refImg_I_Ind]);
					refImg_I_Ind++;

					refImgSum_I_temp += refImg_I[refImg_I_Ind -1];
					printf("refImg_I[refImg_I_Ind-1]: %f\n", refImg_I[refImg_I_Ind-1]);
					refImgSum_I_I_temp += refImg_I[refImg_I_Ind -1] * refImg_I[refImg_I_Ind - 1];

					//printf("sum_I_Iprime_row: %f\n", sum_I_Iprime_row);
					//sum_I_Iprime_row += (Iprime * xxx);
					//printf("Iprime: %f, depth: %f, imageId: %d, col_prime/z + 0.5f: %f, row_prime/z + 0.5f:%f \n", Iprime, depth, imageId, col_prime/z + 0.5f, row_prime/z + 0.5f);
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
			//printf("sum_Iprime: %f, sum_Iprime_Iprime: %f, sum_I_Iprime: %f\n", sum_Iprime, sum_Iprime_Iprime, sum_I_Iprime);
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
	printf("refImgSum_I_temp: %f, refImg_sum_I[threadId]: %f\n", refImgSum_I_temp, refImg_sum_I[threadId]);

//	float cost = (refImg_sum_II[threadId] - refImg_sum_I[threadId] * refImg_sum_I[threadId]/ numOfPixels) 
	//	* (sum_Iprime_Iprime - sum_Iprime * sum_Iprime/ numOfPixels); 
	//float cost = ((refImg_sum_II[threadId]*numOfPixels - refImg_sum_I[threadId] * refImg_sum_I[threadId]) 
	//	* (sum_Iprime_Iprime*numOfPixels - sum_Iprime * sum_Iprime)); 
	float cost1 = refImg_sum_II[threadId] - refImg_sum_I[threadId] * refImg_sum_I[threadId]/ (float)numOfPixels;
	float cost2 = sum_Iprime_Iprime - sum_Iprime * sum_Iprime/ (float)numOfPixels; 
	cost1 = cost1 < 0.00001? 0.0f : cost1;
	cost2 = cost2 < 0.00001? 0.0f : cost2;
	float cost = sqrt(cost1 * cost2);
	//printf("cost1: %f, cost2: %f, cost: %f\n", cost1, cost2, cost);
	//printf("sum_I_Iprime: %f\n", sum_I_Iprime);

	if(cost == 0)
		return 2; // very small color consistency
	else
	{
		//float norminator = sum_I_Iprime * numOfPixels - refImg_sum_I[threadId] * sum_Iprime;
		//return 1 -  abs(norminator)/(cost);
		//printf("xxx: %f, refImg_sum_I[threadId]: %f, numOfPixels: %f\n", (sum_I_Iprime -  refImg_sum_I[threadId] * sum_Iprime/ (float)numOfPixels ), refImg_sum_I[threadId], (float)numOfPixels);
		//printf("sum_I_Iprime: %f, refImg_sum_I[threadId]: %f, sum_Iprime: %f, numOfPixels: %f\n ", sum_I_Iprime, refImg_sum_I[threadId], sum_Iprime, (float)numOfPixels);
		cost = 1.0f - (sum_I_Iprime -  refImg_sum_I[threadId] * sum_Iprime/ (float)numOfPixels )/(cost);
		return cost;
	}
}

template<int WINDOWSIZES> 
inline __device__ float computeNCC(const int &threadId, const float *refImg_I, const float *refImg_sum_I, const float *refImg_sum_II, 
	//const int &imageId, const float &centerRow, const float &centerCol, const float &depth, const int &halfWindowSize, const bool &isRotated, const float& refImgWidth, const float& refImgHeight)
	const int &imageId, const float &rowMinusHalfwindowPlusHalf, const float &colMinusHalfwindowPlusHalf, const float &depth, const int &windowSize, const bool &isRotated, const int &halfWindowSize ,
	const int &refImageWidth, const int &refImageHeight)
	// here the return resutls are 1-NCC, so the range is [0, 2], the smaller value, the better color consistency
{
	float sum_I_Iprime = 0;
	float sum_Iprime_Iprime = 0;
	float sum_Iprime = 0;
	float Iprime;
	
	float transform[9]; 
	float *transformBase = transformHH + 18 * imageId;
	int numOfPixels = 0;
	//for(int i = 0; i<9; i++)
	{
		transform[0] = (transformBase)[0] - (transformBase)[9]/depth;
		transform[1] = (transformBase)[1] - (transformBase)[10]/depth;
		transform[2] = (transformBase)[2] - (transformBase)[11]/depth;
		transform[3] = (transformBase)[3] - (transformBase)[12]/depth;
		transform[4] = (transformBase)[4] - (transformBase)[13]/depth;
		transform[5] = (transformBase)[5] - (transformBase)[14]/depth;
		transform[6] = (transformBase)[6] - (transformBase)[15]/depth;
		transform[7] = (transformBase)[7] - (transformBase)[16]/depth;
		transform[8] = (transformBase)[8] - (transformBase)[17]/depth;
		//transform[i] = (transformHH + 18 * imageId)[i] - (transformHH + 18 * imageId)[i+9]/depth;
	}
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
	float cost1 = refImg_sum_II[threadId] - refImg_sum_I[threadId] * refImg_sum_I[threadId]/ (float)numOfPixels;
	float cost2 = sum_Iprime_Iprime - sum_Iprime * sum_Iprime/ (float)numOfPixels; 
	cost1 = cost1 < 0.00001? 0.0f : cost1;
	cost2 = cost2 < 0.00001? 0.0f : cost2;
	float cost = sqrt(cost1 * cost2);
	if(cost == 0)
		return 2; // very small color consistency
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
inline __device__ void readImageIntoSharedMemory(float *refImg_I, int row, int col, const int& threadId, const bool &isRotated, const int &halfWindowSize) 
	// here assumes N is bigger than HALFBLOCK
{
	// here the init is important. As cols will not go beyond the number of cols. However the shared memory corresponding to that part will be used.

	// the size of the data block: 3N * (2 * halfblock + 1)
	if(!isRotated)
	{
		row -= halfWindowSize;
		//row -= HALFBLOCK;
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
	}
	else
	{
		//row -= HALFBLOCK;
		row -= halfWindowSize;
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
	}
}

template<int WINDOWSIZES>
__device__ void initSharedMemory(float *refImg_I, const int &threadId)
{
	for(int i = 0; i < WINDOWSIZES; i++)
	{

		refImg_I[threadId + i * 3 * N] = 0.0f;
		refImg_I[threadId + i * 3 * N + N] = 0.0f; 
		refImg_I[threadId + i * 3 * N + 2 * N] = 0.0f;
	}
}

template<int WINDOWSIZES>
__global__ void topToDown(bool isFirstStart, float *matchCost, float *refImg, float *refImgI, float *refImgII, int refImgPitch, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated, unsigned int _numOfTargetImages, float SPMAlphaSquare)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int threadId = threadIdx.x;
	extern __shared__ float normalizedSPMap[];
	__shared__ float refImg_I[N *3 * WINDOWSIZES];
	initSharedMemory<WINDOWSIZES>(refImg_I, threadId);

	//if(col < refImageWidth)
	if(col < refImageWidth && blockIdx.x == 18 )
	{
		int windowSize = 2 * halfWindowSize + 1;
		float colMinusHalfwindowPlusHalf = (float)col - halfWindowSize + 0.5f;
		//__shared__ float depth_former_array[N]; // N is number of threads per block 
		//__shared__ float depth_current_array[N]; 
		__shared__ float depth_array[3 * N];
		float * depth_former_array = depth_array;
		float * depth_current_array = depth_array + N;
		float * randDepth = depth_array + 2*N;

//		__shared__ float normalizedSPMap[N * 10u/*TARGETIMGS*/]; // need plus 1 here ****. It seems not necessary
		__shared__ float refImg_sum_I[N];
		__shared__ float refImg_sum_II[N];

//		__shared__ float normalizedSPMap_former[N * TARGETIMGS];
		unsigned int s = (_numOfTargetImages >> 5) + 1; // 5 is because each int type has 32 bits, and divided by 32 is equavalent to shift 5. s is number of bytes used to save selected images
//		__shared__ unsigned int selectedImages[ N * ( /*TARGETIMGS*/ 10 >>5) + N ]; // this is N * s
		unsigned int* selectedImages = (unsigned int*)(normalizedSPMap + N * _numOfTargetImages);

		depth_former_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, 0, col); 	// depth for 1st element
		
		//for(int i = 0; i<TARGETIMGS; i++)
		//	normalizedSPMap_former[i*N + threadId] = accessPitchMemory(SPMap, SPMapPitch, i * refImageHeight + 0, col );

		__shared__ curandState localState[N];		
		if(!isRotated)
			localState[threadId] = *(randState + col);
		else
			localState[threadId] = *((curandState*)((char*)randState + col * randStatePitch));

		float rowMinusHalfwindowPlusHalf = 0.0f - halfWindowSize + 0.5f;
		//--------------------------------------------------------------------------------------------
		//if(isFirstStart)
		//{
		//	for(int row = 0; row < refImageHeight; ++row)
		//	{
		//		refImg_sum_I[threadId] = accessPitchMemory(refImgI, refImgPitch, row, col);
		//		refImg_sum_II[threadId] = accessPitchMemory(refImgII, refImgPitch, row, col);
		//		readImageIntoSharedMemory<WINDOWSIZES>( refImg_I, row, col, threadId, isRotated, halfWindowSize);
		//		depth_current_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col);
		//		float cost1stRow;
		//		for(int imageId = 0; imageId < _numOfTargetImages; imageId ++)
		//		{
		//			cost1stRow = computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_current_array[threadId], windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);
		//			writePitchMemory(matchCost, SPMapPitch,  imageId * refImageHeight, col, cost1stRow);
		//			cost1stRow = exp(-0.5 * cost1stRow * cost1stRow / SPMAlphaSquare);
		//			writePitchMemory(SPMap, SPMapPitch, imageId * refImageHeight, col, cost1stRow); // write SPMap
		//		}
		//		++rowMinusHalfwindowPlusHalf;
		//	}
		//}
		rowMinusHalfwindowPlusHalf = 0.0 - halfWindowSize + 0.5;
		//----------------------------------------------------------------------------------------------

		for( int row = 1; row < refImageHeight; ++row)
		{
			++rowMinusHalfwindowPlusHalf;

			refImg_sum_I[threadId] = accessPitchMemory(refImgI, refImgPitch, row, col);
			refImg_sum_II[threadId] = accessPitchMemory(refImgII, refImgPitch, row, col);
			readImageIntoSharedMemory<WINDOWSIZES>( refImg_I, row, col, threadId, isRotated, halfWindowSize);

			if(row < 5)
				printf("xxxxx:%f\n", tex2DLayered( refImgTexture, row + 0.5f , col + 0.5f, 0));

			depth_current_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col); 
			for(int i = 0; i<s; i++)
				selectedImages[threadId + i * N] = 0;	// initialized to false
			//---------------------------------
			if(numOfSamples == 1)
			{
				for(int i = 0; i<_numOfTargetImages; i++)
					//normalizedSPMap[i*N + threadId] = normalizedSPMap_former[i*N + threadId];
					normalizedSPMap[i * N + threadId ] = accessPitchMemory(SPMap, SPMapPitch, row-1 + i * refImageHeight, col );	// in the first round I only choose 1 sample. And SPMap is chosen from 
			}
			else
			{
				//if(row == refImageHeight - 1)
				{
					for(int i = 0; i<_numOfTargetImages; i++)
						normalizedSPMap[i * N + threadId ] = (accessPitchMemory(SPMap,  SPMapPitch, row + i * refImageHeight, col) 
						+ accessPitchMemory(SPMap, SPMapPitch, row-1 + i * refImageHeight, col) )/2.0f;		// average of the near two
				}
				//else
				//{
				//	for(int i = 0; i<_numOfTargetImages; i++)
				//		normalizedSPMap[i * N + threadId ] = (accessPitchMemory(SPMap,  SPMapPitch, row + i * refImageHeight, col) 
				//		+ accessPitchMemory(SPMap, SPMapPitch, row-1 + i * refImageHeight, col)
				//		+ accessPitchMemory(SPMap, SPMapPitch, row +1 + i * refImageHeight, col)
				//		)/3.0f;		// average of the near two

				//}
				//for(int i = 0; i<_numOfTargetImages; i++)
				//{
				//	normalizedSPMap[i * N + threadId ] = (accessPitchMemory(matchCost,  SPMapPitch, row + i * refImageHeight, col) 
				//		+ accessPitchMemory(matchCost, SPMapPitch, row-1 + i * refImageHeight, col) )/2.0f;		// average of the near two
				//	normalizedSPMap[i * N + threadId ] = exp(-0.5 * normalizedSPMap[i*N + threadId] * normalizedSPMap[i * N + threadId] / (SPMAlphaSquare));
				//}
			}
			//---------------------------------
			for(int i = 1; i<_numOfTargetImages; i++)		
				normalizedSPMap[i * N + threadId] += normalizedSPMap[(i-1) * N + threadId ];
			// normalize
			for(int i = 0; i<_numOfTargetImages; i++)
				normalizedSPMap[i * N + threadId] /= normalizedSPMap[N * (_numOfTargetImages -1) + threadId];

			// draw samples and set the bit to 0
			float numOfTestedSamples = 0;
			float cost[3] = {0.0f};
			// here it is better to generate a random depthmap
			randDepth[threadId] = curand_uniform(&localState[threadId]) * (depthRangeFar - depthRangeNear) + depthRangeNear;

			unsigned int pos;
			for(int j = 0; j < numOfSamples; j++)
			{
				float randNum = curand_uniform(&localState[threadId]);
				/*if(col == 577 && row <= 2 )
				{
					printf("randNum: %f\n", randNum);
				}*/
				int imageId = -1;				
				for(unsigned int i = 0; i < _numOfTargetImages; i++)
				{
					float x = normalizedSPMap[i * N + threadId];
					//if(randNum <= normalizedSPMap[i * N + threadId ])
					
					/*if(col == 577 && row == 2 )
					{
						printf("normalizedSPMap[i*N+threadId]: %f\n", normalizedSPMap[i*N+threadId]);
					}*/

					if(randNum < x)
					{
						unsigned int &stateByte = selectedImages[(i>>5) * N + threadId];
						pos = i - (sizeof(unsigned int) << 3) * (i>>5);  //(i - i /32 * numberOfBitsPerInt)
						if(!CHECK_BIT(stateByte,pos))
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
				if(imageId != -1)
				{
					//if(col == 577 && row == 2 )
					//{
					//	//printf("cost[0]: %f, cost[1]: %f, cost[2]: %f \n", cost[0], cost[1], cost[2]);
					//	//randDepth[threadId] = 7.2;
					//	float xxx = computeNCCNew<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_former_array[threadId], windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);			// accumulate the cost
					//	//printf("row: %d, col: %d, cost[0]: %f, cost[1]: %f, cost[2]: %f, randDepth: %f, depth_former: %f, depth_current: %f \n", row, col, cost[0], cost[1], cost[2], randDepth[threadId], depth_former_array[threadId], depth_current_array[threadId]);
					//	printf("xxx: %f\n", xxx);
					//}	
					cost[0] +=  computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_former_array[threadId], windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);			// accumulate the cost
					if(!isFirstStart)
						cost[1] += accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
					else
						cost[1] +=  computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_current_array[threadId], windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);
					
					cost[2] +=  computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, randDepth[threadId], windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);	
					//cost[2] += cost[1] + 0.1;
					//if(col == 577)
					//	printf("row: %d, col: %d, cost[0]: %f, cost[1]: %f, cost[2]: %f, randDepth: %f, depth_former: %f, depth_current: %f \n", row, col, cost[0], cost[1], cost[2], randDepth[threadId], depth_former_array[threadId], depth_current_array[threadId]);
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
				if(idx != 1 || isFirstStart)
				{	
					if(row == 2 && imageId == 0)
					{
						if(col == 577)
						{
							float sumI = accessPitchMemory(refImgI, refImgPitch, row, col);
							printf("sumI: %f\n", sumI);
							cost[0] = computeNCCNew<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);
						}
						//printf("cost[0]: %f\n", cost[0]);
					}
					cost[0] = computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);
					writePitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col, cost[0]);
				} 
				else
				{
					cost[0] = accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
				}
				cost[0] = exp(-0.5 * cost[0] * cost[0] / (SPMAlphaSquare ));
				/*if(col == 577 && row == 2 )
				{
					printf("SPMap: %f, depth: %f\n", cost[0], bestDepth);
				}	*/

				writePitchMemory(SPMap, SPMapPitch,row + imageId * refImageHeight, col, cost[0]); // write SPMap
			}
		}
		if(!isRotated)
			*(randState + col) = localState[threadId];
		else
			*((curandState*)((char*)randState + col * randStatePitch)) = localState[threadId];
	}
}


template<int WINDOWSIZES>
__global__ void downToTop(bool isFirstStart, float *matchCost, float *refImg, float *refImgI, float *refImgII, int refImgPitch, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated, unsigned int _numOfTargetImages, float SPMAlphaSquare)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int threadId = threadIdx.x;
	extern __shared__ float normalizedSPMap[];

	if(col < refImageWidth)
	{
		int windowSize = halfWindowSize * 2 + 1;
		float colMinusHalfwindowPlusHalf = (float)col - halfWindowSize + 0.5f;
		//__shared__ float depth_former_array[N]; // N is number of threads per block 
		//__shared__ float depth_current_array[N]; 
		__shared__ float depth_array[3 * N];
		float * depth_former_array = depth_array;
		float * depth_current_array = depth_array + N;
		float * randDepth = depth_array + 2*N;
		//__shared__ float sumOfSPMap[N]; 
		//__shared__ float normalizedSPMap[N * /*TARGETIMGS*/ 10u]; // need plus 1 here ****. It seems not necessary

		__shared__ float refImg_sum_I[N];
		__shared__ float refImg_sum_II[N];
		__shared__ float refImg_I[N *3 * WINDOWSIZES];	// WINDOWSIZE SHOULD BE SMALLER LESS THAN N
		

		//__shared__ float normalizedSPMap_former[N * TARGETIMGS];
		unsigned int s = (_numOfTargetImages >> 5) + 1; // 5 is because each int type has 32 bits, and divided by 32 is equavalent to shift 5. s is number of bytes used to save selected images
		//__shared__ unsigned int selectedImages[ N * ( /*TARGETIMGS*/ 10u >>5) + N ]; // this is N * s

		unsigned int* selectedImages = (unsigned int*)(normalizedSPMap + N * _numOfTargetImages);
	
		depth_former_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, refImageHeight - 1, col); 	// depth for 1st element

		__shared__ curandState localState[N];
		if(!isRotated)
			localState[threadId] = *(randState + col);
		else
			localState[threadId] = *((curandState*)((char*)randState + col * randStatePitch));
		float rowMinusHalfwindowPlusHalf = refImageHeight - 1.0f - halfWindowSize + 0.5f;
		for(int row = refImageHeight - 2; row >=0; --row)
		{
			--rowMinusHalfwindowPlusHalf;

			refImg_sum_I[threadId] = accessPitchMemory(refImgI, refImgPitch, row, col);
			refImg_sum_II[threadId] = accessPitchMemory(refImgII, refImgPitch, row, col);
			readImageIntoSharedMemory<WINDOWSIZES>( refImg_I, row, col, threadId, isRotated, halfWindowSize);

			depth_current_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col); 
			for(int i = 0; i<s; i++)
				selectedImages[threadId + i * N] = 0;	// initialized to false
			//---------------------------------
			if(numOfSamples == 1)
			{
				for(int i = 0; i<_numOfTargetImages; i++)
					normalizedSPMap[i * N + threadId ] = accessPitchMemory(SPMap, SPMapPitch, (row + 1) + i * refImageHeight, col );	// in the first round I only choose 1 sample. And SPMap is chosen from 
			//		//normalizedSPMap[i*N + threadId] = normalizedSPMap_former[i*N + threadId];
			}
			else
			{
				//for(int i = 0; i<TARGETIMGS; i++)
				//	normalizedSPMap[i * N + threadId ] = accessPitchMemory(SPMap,  SPMapPitch, row + i * refImageHeight, col) /*/ (sumOfSPMap[threadId] + FLT_MIN )*/;	// devide by 0
				for(int i = 0; i<_numOfTargetImages; i++)
					normalizedSPMap[i * N + threadId ] = (accessPitchMemory(SPMap,  SPMapPitch, row + i * refImageHeight, col) 
						+ accessPitchMemory(SPMap, SPMapPitch, (row + 1) + i * refImageHeight, col) )/2.0f;		// average of the near two
					//normalizedSPMap[i * N + threadId ] = (normalizedSPMap_former[i*N + threadId] 
					//	+ accessPitchMemory(SPMap, SPMapPitch, row + i * refImageHeight, col) )/2.0f;
				//for(int i = 0; i<_numOfTargetImages; i++)
				//{
				//	normalizedSPMap[i * N + threadId ] = (accessPitchMemory(matchCost,  SPMapPitch, row + i * refImageHeight, col) 
				//		+ accessPitchMemory(matchCost, SPMapPitch, row + 1 + i * refImageHeight, col) )/2.0f;		// average of the near two
				//	normalizedSPMap[i * N + threadId ] = exp(-0.5 * normalizedSPMap[i*N + threadId] * normalizedSPMap[i * N + threadId] / (SPMAlphaSquare));
				//}
				//if(row == 0)
				//{
				//	for(int i = 0; i<_numOfTargetImages; i++)
				//		normalizedSPMap[i * N + threadId ] = (accessPitchMemory(SPMap,  SPMapPitch, row + i * refImageHeight, col) 
				//			+ accessPitchMemory(SPMap, SPMapPitch, row+1 + i * refImageHeight, col) )/2.0f;		// average of the near two
				//}
				//else
				//{
				//	for(int i = 0; i<_numOfTargetImages; i++)
				//		normalizedSPMap[i * N + threadId ] = (accessPitchMemory(SPMap,  SPMapPitch, row + i * refImageHeight, col) 
				//		+ accessPitchMemory(SPMap, SPMapPitch, row -1 + i * refImageHeight, col)
				//		+ accessPitchMemory(SPMap, SPMapPitch, row +1 + i * refImageHeight, col)
				//		)/3.0f;		// average of the near two

				//}			
			}

			//---------------------------------
			for(int i = 1; i<_numOfTargetImages; i++)		
				normalizedSPMap[i * N + threadId] += normalizedSPMap[(i-1) * N + threadId ];
			// normalize
			//sumOfSPMap[threadId] = normalizedSPMap[N * (TARGETIMGS - 1) + threadId];
			for(int i = 0; i<_numOfTargetImages; i++)
				//normalizedSPMap[i * N + threadId] /= sumOfSPMap[threadId];
				normalizedSPMap[i * N + threadId] /= normalizedSPMap[N * (_numOfTargetImages -1) + threadId];

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
					{
						unsigned int &stateByte = selectedImages[(i>>5) * N + threadId];
						unsigned int pos = i - (sizeof(int) << 3) * (i>>5);  //(i - i /32 * numberOfBitsPerInt)
						if(!CHECK_BIT(stateByte,pos))
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
				if(imageId != -1)
				{
					cost[0] +=  computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_former_array[threadId], windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);			// accumulate the cost
					cost[1] += accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
					cost[2] +=  computeNCC<WINDOWSIZES>(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, randDepth[threadId], windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);
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
					cost[0] = computeNCC<WINDOWSIZES> (threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated, halfWindowSize, refImageWidth, refImageHeight);
					writePitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col, cost[0]);
				}
				else
				{
					cost[0] = accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
				}
				cost[0] = exp(-0.5 * cost[0] * cost[0] / (SPMAlphaSquare));
				writePitchMemory(SPMap, SPMapPitch,row + imageId * refImageHeight, col, cost[0]); // write SPMap
			}
		}
		if(!isRotated)
			*(randState + col) = localState[threadId];
		else
			*((curandState*)((char*)randState + col * randStatePitch)) = localState[threadId];
	}
}





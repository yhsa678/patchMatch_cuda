#include "patchMatch.h"
#include "cudaTranspose.h"
#include "utility_CUDA.h"
#include "GaussianBlurCUDA.h"
#include  <sstream> 

#define MAX_NUM_IMAGES 128

__global__ void topToDown(bool isFirstStart, float *, float *, float *, float *, int, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated);

__global__ void downToTop(bool isFirstStart, float *, float *, float *, float *, int, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated);

texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> allImgsTexture;
texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> refImgTexture;
__constant__ float transformHH[MAX_NUM_IMAGES * 9 * 2];

#define N 32
#define TARGETIMGS 10u
#define HALFBLOCK 7

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

PatchMatch::PatchMatch( std::vector<Image> &allImage, float nearRange, float farRange, int halfWindowSize, int blockDim_x, int blockDim_y, int refImageId, int numOfSamples): 
	_imageDataBlock(NULL), _allImages_cudaArrayWrapper(NULL), _nearRange(nearRange), _farRange(farRange), _halfWindowSize(halfWindowSize), _blockDim_x(blockDim_x), _blockDim_y(blockDim_y), _refImageId(refImageId),
		_depthMap(NULL), _SPMap(NULL), _psngState(NULL), _depthMapT(NULL), _SPMapT(NULL), _numOfSamples(numOfSamples), _refImage(NULL), _refImageT(NULL)
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

void PatchMatch::run()
{
	int numOfSamples;
	bool isRotated;
	std::cout<< "started" << std::endl;
	CudaTimer t;
	GaussianBlurCUDA gFilter(_refWidth, _refHeight, 2.0f);
	GaussianBlurCUDA gFilterT(_depthMapT->getWidth(), _depthMapT->getHeight(), 2.0f);

	bool isFirstStart = true;
	for(int i = 0; i < 3; i++)
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
		topToDown<<<_gridSize, _blockSize>>>(isFirstStart, _matchCostT->_array2D, _refImageT->_refImageData->_array2D,  _refImageT->_refImage_sum_I->_array2D, _refImageT->_refImage_sum_II->_array2D, _refImageT->_refImage_sum_I->_pitchData,
			_depthMapT->getWidth(), _depthMapT->getHeight(), _depthMapT->_array2D, _depthMapT->_pitchData, 
			_SPMapT->_array2D, _SPMapT->_pitchData, 
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated);
		CudaCheckError();
		gFilterT.FilterMultipleImages( _SPMapT->_array2D, _SPMapT->_pitchData, _SPMapT->getDepth());
		
		//t.stopRecord();
////-----------------------------------------------------------
//	// top to bottom sweep 
		//t.startRecord();
		transposeBackward();
		isFirstStart = false;
		computeCUDAConfig(_depthMap->getWidth(), _depthMap->getHeight(), N, 1);
		isRotated = false;
		topToDown<<<_gridSize, _blockSize>>>(isFirstStart, _matchCost->_array2D, _refImage->_refImageData->_array2D, _refImage->_refImage_sum_I->_array2D, _refImage->_refImage_sum_II->_array2D, _refImage->_refImage_sum_I->_pitchData,
			_depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_array2D, _depthMap->_pitchData, 
			_SPMap->_array2D, _SPMap->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated);
		gFilter.FilterMultipleImages(_SPMap->_array2D, _SPMap->_pitchData, _SPMap->getDepth());

	////////// right to left sweep
		transposeForward();
		computeCUDAConfig(_depthMapT->getWidth(), _depthMapT->getHeight(), N, 1);
		isRotated = true;
		downToTop<<<_gridSize, _blockSize>>>(isFirstStart, _matchCostT->_array2D, _refImageT->_refImageData->_array2D, _refImageT->_refImage_sum_I->_array2D, _refImageT->_refImage_sum_II->_array2D, _refImageT->_refImage_sum_I->_pitchData,
			_depthMapT->getWidth(), _depthMapT->getHeight(), _depthMapT->_array2D, _depthMapT->_pitchData, 
			_SPMapT->_array2D, _SPMapT->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated);
		CudaCheckError();
		gFilterT.FilterMultipleImages( _SPMapT->_array2D, _SPMapT->_pitchData, _SPMapT->getDepth());
		
	//////// bottom to top sweep
		transposeBackward();
		computeCUDAConfig(_depthMap->getWidth(), _depthMap->getHeight(), N, 1);
		isRotated = false;
		downToTop<<<_gridSize, _blockSize>>>(isFirstStart, _matchCost->_array2D, _refImage->_refImageData->_array2D, _refImage->_refImage_sum_I->_array2D, _refImage->_refImage_sum_II->_array2D, _refImage->_refImage_sum_I->_pitchData,
			_depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_array2D, _depthMap->_pitchData, 
			_SPMap->_array2D, _SPMap->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated);
		gFilter.FilterMultipleImages(_SPMap->_array2D, _SPMap->_pitchData, _SPMap->getDepth());
		t.stopRecord();
	}

	_depthMap->saveToFile("depthMap.txt");
	/*for(int i = 0; i< _numOfTargetImages; i++)
	{
		std::stringstream ss; ss<<i;
		std::string fileName = "_SPMap"+ ss.str() + ".txt";
		_SPMap->saveToFile(fileName, i);
	}*/
	std::cout<< "ended " << std::endl;
	// in the end I got the depthmap
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

#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))
#define SET_BIT(var,pos)( (var) |= (1 << (pos) ))

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

inline __device__ float computeNCC(const int &threadId, const float *refImg_I, const float *refImg_sum_I, const float *refImg_sum_II, 
	//const int &imageId, const float &centerRow, const float &centerCol, const float &depth, const int &halfWindowSize, const bool &isRotated, const float& refImgWidth, const float& refImgHeight)
	const int &imageId, const float &rowMinusHalfwindowPlusHalf, const float &colMinusHalfwindowPlusHalf, const float &depth, const int &windowSize, const bool &isRotated)
	// here the return resutls are 1-NCC, so the range is [0, 2], the smaller value, the better color consistency
{
	float sum_I_Iprime = 0;
	float sum_Iprime_Iprime = 0;
	float sum_Iprime = 0;
	float Iprime;

	float transform[9]; 
	float *transformBase = transformHH + 18 * imageId;
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
	int base_refImg_I_Ind = (refImg_I_Ind = N - HALFBLOCK + threadId);

	float sum_Iprime_row;
	float sum_Iprime_Iprime_row;
	float sum_I_Iprime_row; 

	for(int localSharedMemRow = 0; localSharedMemRow < windowSize; localSharedMemRow ++)
	{
		//row += 6.0f;
		//localSharedMemRow += 6.0f;
		//row = max( 0.0f, i) + 0.5f;
		//row = min(refImgHeight - 1.0f, i) + 0.5f;
		//localSharedMemCol = N - HALFBLOCK + threadId;		
		
		//refImg_I_Ind = 3 * N * localSharedMemRow + N - HALFBLOCK + threadId;
	
		//for(float col = centerCol - halfWindowSize; col <= centerCol + halfWindowSize; col++) // x

		sum_Iprime_row = 0.0f;
		sum_Iprime_Iprime_row = 0.0f;
		sum_I_Iprime_row = 0.0f;

		for(int localSharedMemCol = 0; localSharedMemCol < windowSize; localSharedMemCol++)
		{
			// do transform to get the new position
			Iprime = tex2DLayered(allImgsTexture, col_prime/z + 0.5f, row_prime/z + 0.5f, imageId); // textures are not rotated
			//if(Iprime == 1.0f)
				//printf("Iprime is %f, col_prime/z: %f, row_prime/z: %f, z: %f, imageId: %d, isRotated: %d \n", Iprime, col_prime/z + 0.5f, row_prime/z + 0.5f, z, imageId, isRotated);
			sum_Iprime_Iprime_row += (Iprime * Iprime);
			sum_Iprime_row += Iprime;
			//sum_I_Iprime += (Iprime * refImg_I[3*N*localSharedMemRow + localSharedMemCol + N - HALFBLOCK + threadId]);
			sum_I_Iprime_row += (Iprime * refImg_I[refImg_I_Ind++]);

			if(!isRotated)
			{
				//doTransform(col_prime, row_prime, col + 0.5f, row + 0.5f, imageId, transform);
				//doTransform(col_prime, row_prime, (float)localSharedMemCol + centerCol - halfWindowSize + 0.5f, (float)localSharedMemRow + centerRow - halfWindowSize + 0.5f, imageId, transform);
				z += transform[6];
				col_prime += transform[0];
				row_prime += transform[3];
			}
			else
			{
				z += transform[7];
				col_prime += transform[1];
				row_prime += transform[4];
				//doTransform(col_prime, row_prime, row + 0.5f, col + 0.5f, imageId, transform); 
				//doTransform(col_prime, row_prime, (float)localSharedMemRow + centerRow - halfWindowSize + 0.5f, (float)localSharedMemCol + centerCol - halfWindowSize + 0.5f, imageId, transform);
			}
			//refImg_I_Ind++;
		}
		sum_Iprime += sum_Iprime_row;
		sum_Iprime_Iprime += sum_Iprime_Iprime_row;
		sum_I_Iprime += sum_I_Iprime_row;

		refImg_I_Ind = ( base_refImg_I_Ind += (3 * N));

		if(!isRotated) 
		{
			//z =         transform[6] * (centerCol - halfWindowSize + 0.5) + transform[7] * (centerRow - halfWindowSize + 0.5 + localSharedMemRow) + transform[8];
			//col_prime = transform[0] * (centerCol - halfWindowSize + 0.5) + transform[1] * (centerRow - halfWindowSize + 0.5 + localSharedMemRow) + transform[2]; 
		    //row_prime = transform[3] * (centerCol - halfWindowSize + 0.5) + transform[4] * (centerRow - halfWindowSize + 0.5 + localSharedMemRow) + transform[5]; 
			/*z = (base_z + transform[7] * localSharedMemRow);
			col_prime = (base_col_prime + transform[1] * localSharedMemRow); 
			row_prime = (base_row_prime + transform[4] * localSharedMemRow);*/
			z = (base_z += transform[7] );
			col_prime = (base_col_prime += transform[1]); 
			row_prime = (base_row_prime += transform[4]);
			
		}
		else
		{
			//z =         transform[6] * (centerRow - halfWindowSize + 0.5+ localSharedMemRow) + transform[7] * (centerCol - halfWindowSize + 0.5 ) + transform[8];
			//col_prime = transform[0] * (centerRow - halfWindowSize + 0.5+ localSharedMemRow) + transform[1] * (centerCol - halfWindowSize + 0.5 ) + transform[2];
			//row_prime = transform[3] * (centerRow - halfWindowSize + 0.5+ localSharedMemRow) + transform[4] * (centerCol - halfWindowSize + 0.5 ) + transform[5];
			/*z = (base_z + transform[6] * localSharedMemRow);
			col_prime = (base_col_prime + transform[0] * localSharedMemRow); 
			row_prime = (base_row_prime + transform[3] * localSharedMemRow);*/
			z = (base_z += transform[6] );
			col_prime = (base_col_prime += transform[0] ); 
			row_prime = (base_row_prime += transform[3] );
		}
	}	
	//float numOfPixels = static_cast<float>((halfWindowSize * 2 + 1)*(halfWindowSize * 2 + 1));
	float numOfPixels = static_cast<float>(windowSize * windowSize );

//	float cost = (refImg_sum_II[threadId] - refImg_sum_I[threadId] * refImg_sum_I[threadId]/ numOfPixels) 
	//	* (sum_Iprime_Iprime - sum_Iprime * sum_Iprime/ numOfPixels); 
	//float cost = ((refImg_sum_II[threadId]*numOfPixels - refImg_sum_I[threadId] * refImg_sum_I[threadId]) 
	//	* (sum_Iprime_Iprime*numOfPixels - sum_Iprime * sum_Iprime)); 
	float cost1 = refImg_sum_II[threadId] - refImg_sum_I[threadId] * refImg_sum_I[threadId]/ numOfPixels;
	float cost2 = sum_Iprime_Iprime - sum_Iprime * sum_Iprime/ numOfPixels; 
	//if(cost1 < 0.01 || cost2 < 0.01)
	//	printf("cost1: %f, cost2: %f, sum_Iprime_Iprime: %f, sum_Iprime: %f, isequal: %d \n", cost1, cost2, sum_Iprime_Iprime, sum_Iprime, sum_Iprime == 225.0f);
	cost1 = cost1 < 0.00001? 0.0f : cost1;
	cost2 = cost2 < 0.00001? 0.0f : cost2;
	float cost = sqrt(cost1 * cost2);
	if(cost == 0)
		return 2; // very small color consistency
	else
	{
		//float norminator = sum_I_Iprime * numOfPixels - refImg_sum_I[threadId] * sum_Iprime;
		//return 1 -  abs(norminator)/(cost);
		cost = 1 - (sum_I_Iprime -  refImg_sum_I[threadId] * sum_Iprime/numOfPixels )/(cost);
		return cost;
	}
}

inline __device__ int findMinCost(float *cost)
{
	int idx = cost[0] < cost[1]? 0:1;
	idx = cost[idx] < cost[2]? idx:2;
	return idx;
}

//inline __device__ void readImageIntoSharedMemory(const int &threadId, float *refImg_I, float *src, const int &srcPitch, const )
//{
//	for(int i = 0; i< HALFBLOCK * 2 + 1; i++)
//	{				
//		refImg_I[threadId + i * 3 * N] = 0;
//		refImg_I[threadId + i * 3 * N + N] = 0;
//		refImg_I[threadId + i * 3 * N + 2* N] = 0;
//	}
//
//	for(int i = 0; i< HALFBLOCK * 2 + 1; i ++)
//	{
//		if(row + i - HALFBLOCK> 0 )
//		{
//			refImg_I[threadId + i * 3 * N] =  accessPitchMemory(src, srcPitch, row + i - HALFBLOCK, col);
//			refImg_I[threadId + i * 3 * N + N] = accessPitchMemory(
//
//		}
//	}
//}

inline __device__ void readImageIntoSharedMemory(float *refImg_I, int row, int col, const int& threadId, bool isRotated) 
	// here assumes N is bigger than HALFBLOCK
{
	// the size of the data block: 3N * (2 * halfblock + 1)
	if(!isRotated)
	{
		row -= HALFBLOCK;
		col -= N;
		for(int i = 0; i < HALFBLOCK * 2 + 1; i++)
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
		row -= HALFBLOCK;
		col -= N;
		for(int i = 0; i < HALFBLOCK * 2 + 1; i++)
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

__global__ void topToDown(bool isFirstStart, float *matchCost, float *refImg, float *refImgI, float *refImgII, int refImgPitch, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int threadId = threadIdx.x;

	if(col < refImageWidth)
	{
		int windowSize = 2 * halfWindowSize + 1;
		float colMinusHalfwindowPlusHalf = (float)col - halfWindowSize + 0.5f;
		//__shared__ float depth_former_array[N]; // N is number of threads per block 
		//__shared__ float depth_current_array[N]; 
		__shared__ float depth_array[3 * N];
		float * depth_former_array = depth_array;
		float * depth_current_array = depth_array + N;
		float * randDepth = depth_array + 2*N;

		__shared__ float normalizedSPMap[N * TARGETIMGS]; // need plus 1 here ****. It seems not necessary
		__shared__ float refImg_sum_I[N];
		__shared__ float refImg_sum_II[N];
		__shared__ float refImg_I[N *3 * (1 + 2 * HALFBLOCK)];

//		__shared__ float normalizedSPMap_former[N * TARGETIMGS];
		unsigned int s = (TARGETIMGS >> 5) + 1; // 5 is because each int type has 32 bits, and divided by 32 is equavalent to shift 5. s is number of bytes used to save selected images
		__shared__ unsigned int selectedImages[ N * ( TARGETIMGS >>5) + N ]; // this is N * s
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
		if(isFirstStart)
		{
			refImg_sum_I[threadId] = accessPitchMemory(refImgI, refImgPitch, 0, col);
			refImg_sum_II[threadId] = accessPitchMemory(refImgII, refImgPitch, 0, col);
			readImageIntoSharedMemory( refImg_I, 0, col, threadId, isRotated);
			for(int imageId = 0; imageId < TARGETIMGS; imageId ++)
			{
				float cost1stRow = computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_former_array[threadId], windowSize, isRotated);
				writePitchMemory(matchCost, SPMapPitch,  imageId * refImageHeight, col, cost1stRow);
				cost1stRow = exp(-0.5 * cost1stRow * cost1stRow / (0.2 * 0.2));
				writePitchMemory(SPMap, SPMapPitch, imageId * refImageHeight, col, cost1stRow); // write SPMap
			}
		}
		//----------------------------------------------------------------------------------------------

		for( int row = 1; row < refImageHeight; ++row)
		{
			++rowMinusHalfwindowPlusHalf;

			refImg_sum_I[threadId] = accessPitchMemory(refImgI, refImgPitch, row, col);
			refImg_sum_II[threadId] = accessPitchMemory(refImgII, refImgPitch, row, col);
			readImageIntoSharedMemory( refImg_I, row, col, threadId, isRotated);

			depth_current_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col); 
			for(int i = 0; i<s; i++)
				selectedImages[threadId + i * N] = 0;	// initialized to false
			//---------------------------------
			if(numOfSamples == 1)
			{
				for(int i = 0; i<TARGETIMGS; i++)
					//normalizedSPMap[i*N + threadId] = normalizedSPMap_former[i*N + threadId];
					normalizedSPMap[i * N + threadId ] = accessPitchMemory(SPMap, SPMapPitch, row-1 + i * refImageHeight, col );	// in the first round I only choose 1 sample. And SPMap is chosen from 
			}
			else
			{
				for(int i = 0; i<TARGETIMGS; i++)
					normalizedSPMap[i * N + threadId ] = (accessPitchMemory(SPMap,  SPMapPitch, row + i * refImageHeight, col) 
						+ accessPitchMemory(SPMap, SPMapPitch, row-1 + i * refImageHeight, col) )/2.0f;		// average of the near two
			}
			//---------------------------------
			for(int i = 1; i<TARGETIMGS; i++)		
				normalizedSPMap[i * N + threadId] += normalizedSPMap[(i-1) * N + threadId ];
			// normalize
			for(int i = 0; i<TARGETIMGS; i++)
				normalizedSPMap[i * N + threadId] /= normalizedSPMap[N * (TARGETIMGS -1) + threadId];

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
				for(unsigned int i = 0; i < TARGETIMGS; i++)
				{
					if(randNum <= normalizedSPMap[i * N + threadId ])
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
					cost[0] +=  computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_former_array[threadId], windowSize, isRotated);			// accumulate the cost
					if(!isFirstStart)
						cost[1] += accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
					else
						cost[1] +=  computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_current_array[threadId], windowSize, isRotated);
					
					cost[2] +=  computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, randDepth[threadId], windowSize, isRotated);	
					//cost[2] += cost[1] + 0.1;
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
			
			for(int imageId = 0; imageId < TARGETIMGS; imageId ++)
			{
				if(idx != 1 || isFirstStart)
				{	
					cost[0] = computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated);
					//computeNCCTest(testValue, threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated);
					
					//if(blockIdx.x == 0 && threadIdx.x == 0 && imageId == 0)
					//	printf("cost[0]: %f\n", cost[0]);
					writePitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col, cost[0]);
					//float a = accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
				} 
				else
				{

					cost[0] = accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
					//cost[0] = computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated);
					float d = computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated);
					//if(abs(cost[0] - d) > 0.05)
					//	printf("cost[0]: %f, d: %f, threadId: %d, blockIdx: %d, imageId: %d, row: %d, col: %d\n", cost[0] , d, threadId, blockIdx.x, imageId, row, col);
				}
				//writePitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col, cost[0]);
				cost[0] = exp(-0.5 * cost[0] * cost[0] / (0.2 * 0.2));
				writePitchMemory(SPMap, SPMapPitch,row + imageId * refImageHeight, col, cost[0]); // write SPMap
			}

			//float jump = float(imageId * refImageHeight);
			//int baseRow = row;
			//for(int imageId = 0; imageId < TARGETIMGS; imageId++)
			//{
			//	//cost[0] = computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated);
			//	cost[0] = accessPitchMemory(matchCost, SPMapPitch, row, col);
			//	
			//	cost[0] = exp(-0.5 * cost[0] * cost[0] / (0.2 * 0.2));
			//	//normalizedSPMap_former[imageId * N + threadId] = cost[0];
			//	writePitchMemory(SPMap, SPMapPitch, row + imageId * refImageHeight, col, cost[0]); // write SPMap
			//	//writePitchMemory(SPMap, SPMapPitch, baseRow , col, cost[0]); // write SPMap
			//	//baseRow += refImageHeight;
			//}
			
		}
		if(!isRotated)
			*(randState + col) = localState[threadId];
		else
			*((curandState*)((char*)randState + col * randStatePitch)) = localState[threadId];
	}
}

__global__ void downToTop(bool isFirstStart, float *matchCost, float *refImg, float *refImgI, float *refImgII, int refImgPitch, int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int threadId = threadIdx.x;

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
		__shared__ float normalizedSPMap[N * TARGETIMGS]; // need plus 1 here ****. It seems not necessary
		__shared__ float refImg_sum_I[N];
		__shared__ float refImg_sum_II[N];
		__shared__ float refImg_I[N *3 * (1 + 2 * HALFBLOCK)];

		//__shared__ float normalizedSPMap_former[N * TARGETIMGS];
		unsigned int s = (TARGETIMGS >> 5) + 1; // 5 is because each int type has 32 bits, and divided by 32 is equavalent to shift 5. s is number of bytes used to save selected images
		__shared__ unsigned int selectedImages[ N * ( TARGETIMGS >>5) + N ]; // this is N * s
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
			readImageIntoSharedMemory( refImg_I, row, col, threadId, isRotated);

			depth_current_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col); 
			for(int i = 0; i<s; i++)
				selectedImages[threadId + i * N] = 0;	// initialized to false
			//---------------------------------
			if(numOfSamples == 1)
			{
				for(int i = 0; i<TARGETIMGS; i++)
					normalizedSPMap[i * N + threadId ] = accessPitchMemory(SPMap, SPMapPitch, (row + 1) + i * refImageHeight, col );	// in the first round I only choose 1 sample. And SPMap is chosen from 
					//normalizedSPMap[i*N + threadId] = normalizedSPMap_former[i*N + threadId];
			}
			else
			{
				//for(int i = 0; i<TARGETIMGS; i++)
				//	normalizedSPMap[i * N + threadId ] = accessPitchMemory(SPMap,  SPMapPitch, row + i * refImageHeight, col) /*/ (sumOfSPMap[threadId] + FLT_MIN )*/;	// devide by 0
				for(int i = 0; i<TARGETIMGS; i++)
					normalizedSPMap[i * N + threadId ] = (accessPitchMemory(SPMap,  SPMapPitch, row + i * refImageHeight, col) 
						+ accessPitchMemory(SPMap, SPMapPitch, (row + 1) + i * refImageHeight, col) )/2.0f;		// average of the near two
					//normalizedSPMap[i * N + threadId ] = (normalizedSPMap_former[i*N + threadId] 
					//	+ accessPitchMemory(SPMap, SPMapPitch, row + i * refImageHeight, col) )/2.0f;
			}

			//---------------------------------
			for(int i = 1; i<TARGETIMGS; i++)		
				normalizedSPMap[i * N + threadId] += normalizedSPMap[(i-1) * N + threadId ];
			// normalize
			//sumOfSPMap[threadId] = normalizedSPMap[N * (TARGETIMGS - 1) + threadId];
			for(int i = 0; i<TARGETIMGS; i++)
				//normalizedSPMap[i * N + threadId] /= sumOfSPMap[threadId];
				normalizedSPMap[i * N + threadId] /= normalizedSPMap[N * (TARGETIMGS -1) + threadId];

			// draw samples and set the bit to 0
			float numOfTestedSamples = 0.0f;
			float cost[3] = {0.0f};

			// here it is better to generate a random depthmap
			randDepth[threadId] = curand_uniform(&localState[threadId]) * (depthRangeFar - depthRangeNear) + depthRangeNear;
	
			for(int j = 0; j < numOfSamples; j++)
			{
				float randNum = curand_uniform(&localState[threadId]); 

				int imageId = -1;				
				for(int i = 0; i < TARGETIMGS; i++)
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
					//cost[0] +=  computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, (float)row, (float)col, depth_former_array[threadId], halfWindowSize, isRotated, (float)refImageWidth, (float)refImageHeight);			// accumulate the cost
					cost[0] +=  computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_former_array[threadId], windowSize, isRotated);			// accumulate the cost
					//cost[1] +=  computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_current_array[threadId], windowSize, isRotated);
					cost[1] += accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);

					//float a = accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
					//float b = computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II, imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, depth_current_array[threadId], windowSize, isRotated);
					//if(abs(a - b) > 0.025)
						//printf("diff: %f", abs(a-b));

					cost[2] +=  computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, randDepth[threadId], windowSize, isRotated);
				}
			}	
			// find the minimum cost id, and then put cost into global memory 
			numOfTestedSamples = 1.0f/numOfTestedSamples;
			cost[0] *= numOfTestedSamples; cost[1] *= numOfTestedSamples; cost[2] *= numOfTestedSamples;
		
			int idx = findMinCost(cost);
			float bestDepth = depth_array[threadId + N * idx];
			writePitchMemory(depthMap, depthMapPitch, row, col, bestDepth);
			depth_former_array[threadId] = bestDepth;


			for(int imageId = 0; imageId < TARGETIMGS; imageId ++)
			{
				if(idx != 1)
				{
					cost[0] = computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated);
					writePitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col, cost[0]);
				}
				else
				{
					cost[0] = accessPitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col);
				//	cost[0] = computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated);
				//	float d = computeNCC(threadId, refImg_I, refImg_sum_I, refImg_sum_II,imageId, rowMinusHalfwindowPlusHalf, colMinusHalfwindowPlusHalf, bestDepth, windowSize, isRotated);
				//	if(abs(cost[0] - d) > 0.01)
				//		printf("cost[0]: %f, d: %f, threadId: %d, blockIdx: %d\n", cost[0] , d, threadId, blockIdx.x);
				}
				//writePitchMemory(matchCost, SPMapPitch, row + imageId * refImageHeight, col, cost[0]);
				cost[0] = exp(-0.5 * cost[0] * cost[0] / (0.2 * 0.2));
				writePitchMemory(SPMap, SPMapPitch,row + imageId * refImageHeight, col, cost[0]); // write SPMap
			}
			
		}
		if(!isRotated)
			*(randState + col) = localState[threadId];
		else
			*((curandState*)((char*)randState + col * randStatePitch)) = localState[threadId];
	}
}





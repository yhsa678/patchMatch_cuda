#include "patchMatch.h"
#include "cudaTranspose.h"
#include "utility_CUDA.h"
#include "GaussianBlurCUDA.h"
#include  <sstream> 

#define MAX_NUM_IMAGES 128

__global__ void topToDown(int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated);

__global__ void downToTop(int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated);

texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> allImgsTexture;
texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> refImgTexture;
__constant__ float transformHH[MAX_NUM_IMAGES * 9 * 2];

#define N 32
#define TARGETIMGS 10u

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
		_depthMap(NULL), _SPMap(NULL), _psngState(NULL), _depthMapT(NULL), _SPMapT(NULL), _numOfSamples(numOfSamples)
{
	_numOfTargetImages = allImage.size() - 1;
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
	
	// upload H matrix
	cudaMemcpyToSymbol("transformHH", _transformHH , sizeof(float) * 18 * _numOfTargetImages, 0, cudaMemcpyHostToDevice);

	// initialize depthmap and SP(selection probability) map
	_depthMap = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y);
	_SPMap = new Array2D_wrapper<float>(_refWidth, _refHeight, _blockDim_x, _blockDim_y, _numOfTargetImages);
	_psngState = new Array2D_psng(_refWidth, _refHeight, _blockDim_x, _blockDim_y);

	_depthMap->randNumGen(_nearRange, _farRange, _psngState->_array2D, _psngState->_pitchData);
	_SPMap->randNumGen(0.0f, 1.0f, _psngState->_array2D, _psngState->_pitchData); 
	//viewData1DDevicePointer( _SPMap->_array2D, 100);

	_depthMapT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, _blockDim_y);
	_SPMapT = new Array2D_wrapper<float>(_refHeight, _refWidth, _blockDim_x, _blockDim_y, _numOfTargetImages);
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

}

void PatchMatch::transpose(Array2D_wrapper<float> *input, Array2D_wrapper<float> *output)
{
	//transpose2dData(float * input, float *output, int width, int height, int pitchInput, int pitchOutput);
	//tp.transpose2dData(input_device + i * inputPitch * height/sizeof(float), output_device + i * outputPitch * width/sizeof(float), width, height,inputPitch, outputPitch );
	//cudaTranspose::transpose2dData( _depthMap->_array2D, _depthMapT->_array2D, _depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_pitchData, _depthMapT->_pitchData);
	
	for(unsigned int d = 0; d < input->getDepth(); d++)
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
}

void PatchMatch::transposeBackward()
{
	transpose(_depthMapT, _depthMap);
	transpose(_SPMapT, _SPMap);
}

void PatchMatch::run()
{
	int numOfSamples;
	bool isRotated;
	std::cout<< "started" << std::endl;
	CudaTimer t;
	GaussianBlurCUDA gFilter(_refWidth, _refHeight, 2.0f);
	GaussianBlurCUDA gFilterT(_depthMapT->getWidth(), _depthMapT->getHeight(), 2.0);

	
	for(int i = 0; i < 3; i++)
	{
	// left to right sweep
//-----------------------------------------------------------
		std::cout<< "Iteration " << i << " starts" << std::endl;		t.startRecord();
		//if(i == 0)
		//	numOfSamples = 1; // ****
		//else
		numOfSamples = _numOfSamples;
		
		transposeForward();
		computeCUDAConfig(_depthMapT->getWidth(), _depthMapT->getHeight(), N, 1);
		isRotated = true;
		topToDown<<<_gridSize, _blockSize>>>(_depthMapT->getWidth(), _depthMapT->getHeight(), _depthMapT->_array2D, _depthMapT->_pitchData, 
			_SPMapT->_array2D, _SPMapT->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated);
		CudaCheckError();
		gFilterT.FilterMultipleImages( _SPMapT->_array2D, _SPMapT->_pitchData, _SPMapT->getDepth());
//-----------------------------------------------------------
	// top to bottom sweep 
		transposeBackward();
		computeCUDAConfig(_depthMap->getWidth(), _depthMap->getHeight(), N, 1);
		isRotated = false;
		topToDown<<<_gridSize, _blockSize>>>(_depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_array2D, _depthMap->_pitchData, 
			_SPMap->_array2D, _SPMap->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated);
		gFilter.FilterMultipleImages(_SPMap->_array2D, _SPMap->_pitchData, _SPMap->getDepth());

	////// right to left sweep
		transposeForward();
		computeCUDAConfig(_depthMapT->getWidth(), _depthMapT->getHeight(), N, 1);
		isRotated = true;
		downToTop<<<_gridSize, _blockSize>>>(_depthMapT->getWidth(), _depthMapT->getHeight(), _depthMapT->_array2D, _depthMapT->_pitchData, 
			_SPMapT->_array2D, _SPMapT->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated);
		gFilterT.FilterMultipleImages( _SPMapT->_array2D, _SPMapT->_pitchData, _SPMapT->getDepth());

		
	////// bottom to top sweep
		transposeBackward();
		computeCUDAConfig(_depthMap->getWidth(), _depthMap->getHeight(), N, 1);
		isRotated = false;
		downToTop<<<_gridSize, _blockSize>>>(_depthMap->getWidth(), _depthMap->getHeight(), _depthMap->_array2D, _depthMap->_pitchData, 
			_SPMap->_array2D, _SPMap->_pitchData,
			numOfSamples, _psngState->_array2D, _psngState->_pitchData, _nearRange, _farRange, _halfWindowSize, isRotated);
		gFilter.FilterMultipleImages(_SPMap->_array2D, _SPMap->_pitchData, _SPMap->getDepth());
		t.stopRecord();
	}
	_depthMap->saveToFile("depthMap.txt");
	for(int i = 0; i< _numOfTargetImages; i++)
	{
		std::stringstream ss; ss<<i;
		std::string fileName = "_SPMap"+ ss.str() + ".txt";
		_SPMap->saveToFile(fileName, i);
	}
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

inline __device__ void doTransform(float *col_prime, float *row_prime, float col, float row, int imageId, float *transform)
{
	//float *base = &transformHH[0] +  18 * imageId;
	//float z = (base[6] - base[15]/depth) * col + (base[7] - base[16]/depth) * row + (base[8] - base[17]/depth);
	//*col_prime = ((base[0] - base[9]/depth) * col + (base[1] - base[10]/depth) * row + (base[2] - base[11]/depth))/z;
	//*row_prime = ((base[3] - base[12]/depth) * col + (base[4] - base[13]/depth) * row + (base[5] - base[14]/depth))/z;
	float z = transform[6] * col + transform[7] * row + transform[8];
	*col_prime = (transform[0] * col + transform[1] * row + transform[2])/z;
	*row_prime = (transform[3] * col + transform[4] * row + transform[5])/z;

}

inline __device__ float computeNCC(const int &imageId, const float &centerRow, const float &centerCol, const float &depth, const int &halfWindowSize, const bool &isRotated, const float& refImgWidth, const float& refImgHeight)
	// here the return resutls are 1-NCC, so the range is [0, 2], the smaller value, the better color consistency
{
	float col_prime;
	float row_prime;

	float sum_I_I = 0;
	float sum_I_Iprime = 0;
	float sum_Iprime_Iprime = 0;
	float sum_I = 0;
	float sum_Iprime = 0;
	
	//float windowSize = 0;
	float col;
	float row;

	float *base = &transformHH[0] +  18 * imageId;
	float transform[9];
#pragma unroll
	for(int i = 0; i<9; i++)
		transform[i] = base[i] - base[i + 9]/depth;

	float I;
	float Iprime;

	for(float i = centerRow - halfWindowSize; i <= centerRow + halfWindowSize; i++) // y
	{
		row = max( 0.0f, i) + 0.5f;
		row = min(refImgHeight - 1.0f, i) + 0.5f;
		for(float j = centerCol - halfWindowSize; j <= centerCol + halfWindowSize; j++) // x
		{
			//if( col < 0 || col> refImgWidth - 1.0f || row< 0 || row> refImgHeight - 1.0f) 
			//	continue;
			//col = j<0            ? 0 : j;
			col = max(0.0f, j) + 0.5f;
			col = min(refImgWidth- 1.0f, j) + 0.5f;

			// do transform to get the new position
			if(!isRotated)
				doTransform(&col_prime, &row_prime, col, row, imageId, transform);
			else
				doTransform(&col_prime, &row_prime, row, col, imageId, transform);
			Iprime = tex2DLayered(allImgsTexture, col_prime + 0.5f, row_prime + 0.5f, imageId); // textures are not rotated
			sum_Iprime_Iprime += (Iprime * Iprime);
			sum_Iprime += Iprime;

	//		Iprime = 0.4;
			if(!isRotated)
				I = tex2DLayered( refImgTexture, col, row, 0);
			else
				I = tex2DLayered( refImgTexture, row, col, 0);
			//I = 0.123;

			sum_I_I += (I * I);
			sum_I_Iprime += (Iprime * I);
			sum_I += I;
			//++windowSize;
		}
	}	
	float windowSize = halfWindowSize * 2.0f + 1.0f;
	windowSize *= windowSize;
	float cost = ((sum_I_I - sum_I * sum_I/windowSize) * (sum_Iprime_Iprime - sum_Iprime * sum_Iprime/windowSize )); 
	cost = cost <=0? 0 : sqrt(cost);
	if(cost == 0)
		return 2; // very small color consistency
	else
	{
		return 1 - (sum_I_Iprime -  sum_I * sum_Iprime/windowSize )/(cost);
	}
}

inline __device__ int findMinCost(float *cost)
{
	int idx = cost[0] < cost[1]? 0:1;
	idx = cost[idx] < cost[2]? idx:2;
	return idx;
}

__global__ void topToDown(int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int threadId = threadIdx.x;

	if(col < refImageWidth)
	{
		__shared__ float depth_former_array[N]; // N is number of threads per block 
		__shared__ float depth_current_array[N]; 
		__shared__ float sumOfSPMap[N]; 
		__shared__ float normalizedSPMap[N * TARGETIMGS]; // need plus 1 here ****. It seems not necessary
		__shared__ float normalizedSPMap_former[N * TARGETIMGS];
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
		for( int row = 1; row < refImageHeight; ++row)
		{
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
					//normalizedSPMap[i * N + threadId ] = (normalizedSPMap_former[i*N + threadId] 
					//	+ accessPitchMemory(SPMap, SPMapPitch, row + i * refImageHeight, col) )/2.0f;
			}
			//---------------------------------
			for(int i = 1; i<TARGETIMGS; i++)		
				normalizedSPMap[i * N + threadId] += normalizedSPMap[(i-1) * N + threadId ];
			// normalize
			sumOfSPMap[threadId] = normalizedSPMap[N * (TARGETIMGS - 1) + threadId];
			for(int i = 0; i<TARGETIMGS; i++)
				normalizedSPMap[i * N + threadId] /= sumOfSPMap[threadId];

			// draw samples and set the bit to 0
			float numOfTestedSamples = 0;
			float cost[3] = {0.0f};

			// here it is better to generate a random depthmap
			float randDepth;
			//if(!isRotated)	
				randDepth = curand_uniform(&localState[threadId]) * (depthRangeFar - depthRangeNear) + depthRangeNear;
				//randDepth = drawRandNum(randState, randStatePitch, col, row, depthRangeNear, depthRangeFar);
			//else
				//float randNum = curand_uniform(&localState) * (depthRangeFar - depthRangeNear) + depthRangeNear;
				//randDepth = drawRandNum(randState, randStatePitch, row, col, depthRangeNear, depthRangeFar);
			unsigned int pos;
			for(int j = 0; j < numOfSamples; j++)
			{
				float randNum; 
				//if(!isRotated)
					//randNum = drawRandNum(randState, randStatePitch, col, row, 0.0f, 1.0f);				
					randNum = curand_uniform(&localState[threadId]);
				//else
				//	randNum = drawRandNum(randState, randStatePitch, row, col, 0.0f, 1.0f);

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
					cost[0] +=  computeNCC(imageId, (float)row, (float)col, depth_former_array[threadId], halfWindowSize, isRotated, (float)refImageWidth, (float)refImageHeight);			// accumulate the cost
					cost[1] +=  computeNCC(imageId, (float)row, (float)col, depth_current_array[threadId], halfWindowSize, isRotated, (float)refImageWidth, (float)refImageHeight);
					cost[2] +=  computeNCC(imageId, (float)row, (float)col, randDepth, halfWindowSize, isRotated, (float)refImageWidth, (float)refImageHeight);
					//cost[2] += cost[1] + 0.1;
				}
			}	
			// find the minimum cost id, and then put cost into global memory 
			cost[0] /= numOfTestedSamples; cost[1] /= numOfTestedSamples; cost[2] /= numOfTestedSamples;
		
			int idx = findMinCost(cost);
			float bestDepth;
			if(idx == 0)
			{
				writePitchMemory(depthMap, depthMapPitch, row, col, depth_former_array[threadId] );
				bestDepth = depth_former_array[threadId];
			}
			else if(idx ==1)
			{
				writePitchMemory(depthMap, depthMapPitch, row, col, depth_current_array[threadId] );
				bestDepth = depth_current_array[threadId];
			}
			else
			{
				writePitchMemory(depthMap, depthMapPitch, row, col, randDepth );
				bestDepth = randDepth;
			}
			// Here I need to calculate SPMap based on the best depth, and put it into SPMap

			float variance_inv = 1.0/(0.2 * 0.2);
			//if(idx != 1 || numOfSamples == 1)
			for(int imageId = 0; imageId < TARGETIMGS; imageId++)
			{
				cost[0] = computeNCC(imageId, (float)row, (float)col, bestDepth, halfWindowSize, isRotated, (float)refImageWidth, (float)refImageHeight);
				cost[0] = exp(-0.5 * cost[0] * cost[0] * variance_inv);
				//normalizedSPMap_former[imageId * N + threadId] = cost[0];
				writePitchMemory(SPMap, SPMapPitch, (float)row + imageId * refImageHeight, (float)col, cost[0]); // write SPMap
			}
			// write depth
			writePitchMemory(depthMap, depthMapPitch, (float)row, (float)col, bestDepth);
			// swap depth former and depth current
			depth_former_array[threadId] = bestDepth;
		}		if(!isRotated)
			*(randState + col) = localState[threadId];
		else
			*((curandState*)((char*)randState + col * randStatePitch)) = localState[threadId];
	}
}

__global__ void downToTop(int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int threadId = threadIdx.x;

	if(col < refImageWidth)
	{
		__shared__ float depth_former_array[N]; // N is number of threads per block 
		__shared__ float depth_current_array[N]; 
		__shared__ float sumOfSPMap[N]; 
		__shared__ float normalizedSPMap[N * TARGETIMGS]; // need plus 1 here ****. It seems not necessary
		__shared__ float normalizedSPMap_former[N * TARGETIMGS];
		unsigned int s = (TARGETIMGS >> 5) + 1; // 5 is because each int type has 32 bits, and divided by 32 is equavalent to shift 5. s is number of bytes used to save selected images
		__shared__ unsigned int selectedImages[ N * ( TARGETIMGS >>5) + N ]; // this is N * s
		depth_former_array[threadId] = accessPitchMemory(depthMap, depthMapPitch, refImageHeight - 1, col); 	// depth for 1st element

		//for(int i = 0; i<TARGETIMGS; i++)
		//	normalizedSPMap_former[i*N + threadId] = accessPitchMemory(SPMap, SPMapPitch, i * refImageHeight + refImageHeight - 1, col );

		__shared__ curandState localState[N];
		if(!isRotated)
			localState[threadId] = *(randState + col);
		else
			localState[threadId] = *((curandState*)((char*)randState + col * randStatePitch));

		for(int row = refImageHeight - 2; row >=0; --row)
		{
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
			sumOfSPMap[threadId] = normalizedSPMap[N * (TARGETIMGS - 1) + threadId];
			for(int i = 0; i<TARGETIMGS; i++)
				normalizedSPMap[i * N + threadId] /= sumOfSPMap[threadId];

			// draw samples and set the bit to 0
			float numOfTestedSamples = 0;
			float cost[3] = {0.0f};

			// here it is better to generate a random depthmap
			float randDepth = curand_uniform(&localState[threadId]) * (depthRangeFar - depthRangeNear) + depthRangeNear;
		/*	float randDepth;
			if(!isRotated)	
				randDepth = drawRandNum(randState, randStatePitch, col, row, depthRangeNear, depthRangeFar);
			else
				randDepth = drawRandNum(randState, randStatePitch, row, col, depthRangeNear, depthRangeFar);
*/
			for(int j = 0; j < numOfSamples; j++)
			{
				float randNum = curand_uniform(&localState[threadId]); 
				/*if(!isRotated)
					randNum = drawRandNum(randState, randStatePitch, col, row, 0.0f, 1.0f);				
				else
					randNum = drawRandNum(randState, randStatePitch, row, col, 0.0f, 1.0f);*/

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
					//computeNCC(int imageId, float centerRow, float centerCol, float depth, int halfWindowSize);
					cost[0] +=  computeNCC(imageId, (float)row, (float)col, depth_former_array[threadId], halfWindowSize, isRotated, (float)refImageWidth, (float)refImageHeight);			// accumulate the cost
					cost[1] +=  computeNCC(imageId, (float)row, (float)col, depth_current_array[threadId], halfWindowSize, isRotated, (float)refImageWidth, (float)refImageHeight);
					cost[2] +=  computeNCC(imageId, (float)row, (float)col, randDepth, halfWindowSize, isRotated, (float)refImageWidth, (float)refImageHeight);
				}
			}			
			// find the minimum cost id, and then put cost into global memory 
			cost[0] /= numOfTestedSamples; cost[1] /= numOfTestedSamples; cost[2] /= numOfTestedSamples;
		
			int idx = findMinCost(cost);
			float bestDepth;
			if(idx == 0)
			{
				writePitchMemory(depthMap, depthMapPitch, row, col, depth_former_array[threadId] );
				bestDepth = depth_former_array[threadId];
			}
			else if(idx ==1)
			{
				writePitchMemory(depthMap, depthMapPitch, row, col, depth_current_array[threadId] );
				bestDepth = depth_current_array[threadId];
			}
			else
			{
				writePitchMemory(depthMap, depthMapPitch, row, col, randDepth );
				bestDepth = randDepth;
			}
			// Here I need to calculate SPMap based on the best depth, and put it into SPMap

			float variance_inv = 1.0/(0.2 * 0.2);
			//if(idx != 1 || numOfSamples == 1)
				for(int imageId = 0; imageId < TARGETIMGS; imageId++)
				{
					cost[0] = computeNCC(imageId, (float)row, (float)col, bestDepth, halfWindowSize, isRotated, (float)refImageWidth, (float)refImageHeight);
					cost[0] = exp(-0.5 * cost[0] * cost[0] * variance_inv);
					//normalizedSPMap_former[imageId * N + threadId] = cost[0];
					writePitchMemory(SPMap, SPMapPitch, (float)row + imageId * refImageHeight, (float)col, cost[0]);
				}
			// write depth
			writePitchMemory(depthMap, depthMapPitch, (float)row, (float)col, bestDepth);
			// swap depth former and depth current
			depth_former_array[threadId] = bestDepth;
		}

		if(!isRotated)
			*(randState + col) = localState[threadId];
		else
			*((curandState*)((char*)randState + col * randStatePitch)) = localState[threadId];
	}
}





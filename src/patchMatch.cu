#include "patchMatch.h"
#include "cudaTranspose.h"

#define MAX_NUM_IMAGES 128

texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> allImgsTexture;
texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> refImgTexture;
__constant__ float transformHH[MAX_NUM_IMAGES * 9 * 2];

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
					memcpy( (void *)dest, (void *)source, allImage[i]._imageData.rows * allImage[i]._imageData.cols * numOfChannels * sizeof(unsigned char));	

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
			memcpy((void*)dest, (void*)source,  _refWidth * _refHeight * numOfChannels * sizeof(unsigned char) );	
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
			memcpy((void*)(_transformHH+offset), (void*)allImage[i]._H1.data, 18 * sizeof(float));
			offset += 18;
			memcpy((void*)(_transformHH+offset), (void*)allImage[i]._H2.data, 18 * sizeof(float));
			offset += 18;
		}
	}

} 

PatchMatch::PatchMatch( std::vector<Image> &allImage, float nearRange, float farRange, int halfWindowSize, int blockDim_x, int blockDim_y, int refImageId): 
	_imageDataBlock(NULL), _allImages_cudaArrayWrapper(NULL), _nearRange(nearRange), _farRange(farRange), _halfWindowSize(halfWindowSize), _blockDim_x(blockDim_x), _blockDim_y(blockDim_y), _refImageId(refImageId),
		_depthMap(NULL), _SPMap(NULL), _psngState(NULL), _depthMapT(NULL), _SPMapT(NULL)
{
	_numOfTargetImages = allImage.size() - 1;
	if(_numOfTargetImages == 0)
	{
		std::cout<< "Error: there is no images" << std::endl;
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
	allImgsTexture.addressMode[0] = cudaAddressModeWrap; allImgsTexture.addressMode[1] = cudaAddressModeClamp; 
	allImgsTexture.filterMode = cudaFilterModeLinear;	allImgsTexture.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(allImgsTexture, _allImages_cudaArrayWrapper->_array3D));	// bind to texture	
	
	refImgTexture.addressMode[0] = cudaAddressModeWrap; refImgTexture.addressMode[1] = cudaAddressModeClamp; 
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
	
	for(unsigned int d = 0; d < _depthMap->getDepth(); d++)
	{
		cudaTranspose::transpose2dData( input->_array2D + d * input->_pitchData * input->getHeight(),
										output->_array2D + d * output->_pitchData * output->getHeight(),
										input->getWidth(), input->getHeight(), output->_pitchData, output->_pitchData);
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

	for(int i = 0; i<3; i++)
	{
	// left to right sweep
//-----------------------------------------------------------
		transposeForward();

//-----------------------------------------------------------
	// top to bottom sweep 
		transposeBackward();

	// right to left sweep
		transposeForward();

	// bottom to top sweep

		transposeBackward();
	}

	// in the end I got the depthmap

}

#define N 32
#define TARGETIMGS 20
#define NUMOFSAMPLES 4

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

inline __device__ void doTransform(float *col_prime, float *row_prime, float col, float row, int imageId, float depth)
{
	float *base = &transformHH[0] +  18 * imageId ;
	float z = (base[6] - base[15]/depth) * col + (base[7] - base[16]/depth) * row + (base[8] - base[17]/depth);
	*col_prime = ((base[0] - base[9]/depth) * col + (base[1] - base[10]/depth) * row + (base[2] - base[11]/depth))/z;
	*row_prime = ((base[3] - base[12]/depth) * col + (base[4] - base[13]/depth) * row + (base[5] - base[14]/depth))/z;

}


inline __device__ float computeNCC(int imageId, float centerRow, float centerCol, float depth, int halfWindowSize, bool isRotated)
	// here the return resutls are 1-NCC, so the range is [0, 2], the smaller value, the better color consistency
{
	float col_prime;
	float row_prime;

	float sum_I_I = 0;
	float sum_I_Iprime = 0;
	float sum_Iprime_Iprime = 0;
	float sum_I = 0;
	float sum_Iprime = 0;

	for(float row = centerRow - halfWindowSize; row <= centerRow + halfWindowSize; row++) // y
	{
		for(float col = centerCol - halfWindowSize; col <= centerCol + halfWindowSize; col++) // x
		{
			float I = tex2DLayered( refImgTexture , col + 0.5f , row + 0.5f,  imageId);

			// do transform to get the new position
			if(!isRotated)
				doTransform(&col_prime, &row_prime, col + 0.5f, row + 0.5f, imageId, depth);
			else
				doTransform(&col_prime, &row_prime, row + 0.5f, col + 0.5f, imageId, depth);

			float Iprime = tex2DLayered(allImgsTexture, col_prime, row_prime, imageId); // textures are not rotated

			sum_I_I += (I * I);
			sum_Iprime_Iprime += (Iprime * Iprime);
			sum_I_Iprime += (Iprime * I);
			sum_I += I;
			sum_Iprime += Iprime;
		}
	}	
	float cost = (sum_I_I - 1/(2*halfWindowSize + 1) * sum_I * sum_I) * (sum_Iprime_Iprime - 1/(2*halfWindowSize + 1) * sum_Iprime * sum_Iprime );

	if(cost == 0)
		return 2; // very small color consistency
	else
		return 1 - (sum_I_Iprime - 1/(2*halfWindowSize + 1) * sum_I * sum_Iprime )/cost;
}

inline __device__ int findMinCost(float *cost)
{
	int idx = cost[0] < cost[1]? 0:1;
	cost[1] = cost[idx];
	idx = cost[1] < cost[2]? 1:2;
	return idx;
}

__global__ void topToDown(int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize, bool isRotated)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int threadId = threadIdx.x;

	if(col < refImageWidth)
	{
		__shared__ float depth_former_array[N]; 
		__shared__ float depth_current_array[N]; 
		__shared__ float sumOfSPMap[N]; 
		__shared__ float normalizedSPMap[N * TARGETIMGS]; // need plus 1 here ****
		int s = (TARGETIMGS)>>5 + 1; // 5 is because each int type has 32 bits, and divided by 32 is equavalent to shift 5. s is number of bytes used to save selected images
		float * depth_former = &depth_former_array[0];
		float * depth_current = &depth_current_array[0];
		__shared__ int selectedImages[ N * ( TARGETIMGS >>5) + N ]; // this is N * s

		depth_former[threadId] = accessPitchMemory(depthMap, depthMapPitch, 0, col); 	// depth for 1st element
		for( int row = 1; row < refImageHeight; ++row)
		{
			depth_current[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col); 
			//sumOfSPMap[threadId] = 0;
			
			//if(threadId < N * s) // fix N = 32, it equals number of threads per block
			for(int i = 0; i<s; i++)
				selectedImages[threadId + i * N] = 0;	// initialized to false
			//__syncthreads();	// Here to syncthreads as we need to calculate 
			//---------------------------------
			//for(int i = 0; i<TARGETIMGS; i++)
			//	sumOfSPMap[threadId] += accessPitchMemory(SPMap,  SPMapPitch, row * TARGETIMGS + i, col);
			for(int i = 0; i<TARGETIMGS; i++)
				normalizedSPMap[i * N + threadId ] = accessPitchMemory(SPMap,  SPMapPitch, row + TARGETIMGS * refImageHeight, col) /*/ (sumOfSPMap[threadId] + FLT_MIN )*/;	// devide by 0
			
			for(int i = 1; i<TARGETIMGS; i++)		//**** accumulate?
				normalizedSPMap[threadId * i] += normalizedSPMap[threadId * (i-1)];
			// normalize
			sumOfSPMap[threadId] = normalizedSPMap[N * (TARGETIMGS - 1) + threadId];
			for(int i = 0; i<TARGETIMGS; i++)
				normalizedSPMap[i * N + threadId] /= sumOfSPMap[threadId];

			// draw samples and set the bit to 0
			float numOfTestedSamples = 0;
			float cost[3] = {0.0f};
			float randDepth = drawRandNum(randState, randStatePitch, col, row, depthRangeNear, depthRangeFar);
			for(int j = 0; j < numOfSamples; j++)
			{
				float randNum = drawRandNum(randState, randStatePitch, col, row, 0.0f, 1.0f);				
				int imageId = -1;				
				for(int i = 0; i < TARGETIMGS; i++)
				{
					if(randNum < normalizedSPMap[i + threadId * N])
					{
						int stateByte = selectedImages[(i>>5) * N + threadId];
						int pos = i - (sizeof(int) << 3) * (i>>5);  //(i - i /32 * 32)
						// check the bit first and set imageId
						bool state = CHECK_BIT(stateByte,pos);
						if(!state) // if not set, then the images are not tested before. So 
						{
							imageId = i;
							numOfTestedSamples++;
						}
						// then set the bit
						SET_BIT(stateByte, pos);
						break;
					}
				}
				// image id is i( id != -1). Test the id using NCC, with 3 different depth. 
				if(imageId != -1)
				{
					//computeNCC(int imageId, float centerRow, float centerCol, float depth, int halfWindowSize);
					cost[0] +=  computeNCC(imageId, (float)row, (float)col, depth_former[threadId], halfWindowSize, isRotated);
					cost[1] +=  computeNCC(imageId, (float)row, (float)col, depth_current[threadId], halfWindowSize, isRotated);
					cost[2] +=  computeNCC(imageId, (float)row, (float)col, randDepth, halfWindowSize, isRotated);
				}
			}			
			// find the minimum cost id, and then put cost into global memory 
			cost[0] /= numOfTestedSamples; cost[1] /= numOfTestedSamples; cost[2] /= numOfTestedSamples;
		
			int idx = findMinCost(cost);
			float bestDepth;
			if(idx == 0)
			{
				writePitchMemory(depthMap, depthMapPitch, row, col, depth_former[threadId] );
				bestDepth = depth_former[threadId];
			}
			else if(idx ==1)
			{
				writePitchMemory(depthMap, depthMapPitch, row, col, depth_current[threadId] );
				bestDepth = depth_current[threadId];
			}
			else
			{
				writePitchMemory(depthMap, depthMapPitch, row, col, randDepth );
				bestDepth = randDepth;
			}
			// Here I need to calculate SPMap based on the best depth, and put it into SPMap

			float variance_inv = 1.0/(0.2 * 0.2);
			for(int imageId = 0; imageId < TARGETIMGS; imageId++)
			{
				cost[0] = computeNCC(imageId, (float)row, (float)col, bestDepth, halfWindowSize, isRotated);
				cost[0] = exp(-0.5 * cost[0] * cost[0] * variance_inv);
				writePitchMemory(SPMap, SPMapPitch, (float)row + imageId * refImageHeight, (float)col, cost[0]);
			}

			// swap depth former and depth current
			if(threadId == 0)
			{
				float *tempAddr = depth_former;
				depth_former = depth_current;
				depth_current = tempAddr;
			}
		}
	}
}



__global__ void testTextureArray_kernel()
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	//printf("hello world. threadId: %d\n", id);
	//test it
	float fetchedData = 0;

	for(int j = 0; j < 1000; j++)
		for(int i = 0; i < 49; i++)
		{
			int layer = id/3;
			fetchedData += tex2DLayered( allImgsTexture, id, i + 0.5,  layer);
	
			//fetchedData += tex2DLayered( allImgsTexture, id, i  + 0.5,  layer);
	
			//fetchedData += tex2DLayered( allImgsTexture, id, i  + 0.5,  layer);
		}
	/*for(int j = 0; j < 1000; j++)
		for(int i = 0; i < 49; i++)
		{
			fetchedData += tex2DLayered( allImgsTexture, id + 0.5, i + 0.5,  1);		
		}
	for(int j = 0; j < 1000; j++)
		for(int i = 0; i < 49; i++)
		{
			fetchedData += tex2DLayered( allImgsTexture, id + 0.5, i + 0.5,  2);		
		}*/

	if( id == 0)
		printf("data: %f\n", fetchedData);
	
}

void PatchMatch::testTextureArray()
{
	dim3 blocksize(32,1,1);
	dim3 gridsize(15,1,1);
	CudaTimer t;
	t.startRecord();
	testTextureArray_kernel<<<gridsize, blocksize>>>();
	t.stopRecord();
	CudaCheckError();

}



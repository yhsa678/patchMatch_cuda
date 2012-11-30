#include "patchMatch.h"

texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> allImgsTexture;
texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> refImgTexture;

PatchMatch::PatchMatch(const std::vector<Image> &allImage, float nearRange, float farRange, int halfWindowSize, int blockDim_x, int blockDim_y): 
	_imageDataBlock(NULL), _allImages_cudaArrayWrapper(NULL), _nearRange(nearRange), _farRange(farRange), _halfWindowSize(halfWindowSize),
		_blockDim_x(blockDim_x), _blockDim_y(blockDim_y)
{
	_numOfImages = allImage.size();
	if(_numOfImages == 0)
	{
		std::cout<< "Error: there is no images" << std::endl;
		exit(EXIT_FAILURE);
	}

	// find maximum size of each dimension
	int numOfChannels =  allImage[0]._imageData.channels();
	int maxWidth = allImage[0]._imageData.cols; 
	int maxHeight = allImage[0]._imageData.rows;
	for(unsigned int i = 1; i < allImage.size(); i++)
	{
		if(allImage[i]._imageData.cols > maxWidth)
		{
			maxWidth = allImage[i]._imageData.cols;
		}
		if(allImage[i]._imageData.rows > maxHeight)
		{
			maxHeight = allImage[i]._imageData.rows;
		}
	}
	// ---------- assign memory, copy data
	int sizeOfBlock = maxWidth * numOfChannels * maxHeight  * _numOfImages;
	_imageDataBlock = new unsigned char[sizeOfBlock]();
	// copy row by row
	//char *dest = _imageDataBlock;
	for(unsigned int i = 0; i < allImage.size(); i++)
	{	
		unsigned char *dest = _imageDataBlock + (maxWidth * numOfChannels * maxHeight) * i;
		unsigned char *source = allImage[i]._imageData.data;

		for( int j = 0; j < maxHeight; j++)
		{
			if(j < allImage[i]._imageData.rows)
			{
				memcpy( (void *)dest, (void *)source, allImage[i]._imageData.step * sizeof(unsigned char));	

				dest += (maxWidth * numOfChannels);
				source += allImage[i]._imageData.step;
			}			
		}
	}
	
	// ---------- initialize array
	_allImages_cudaArrayWrapper = new CudaArray_wrapper(maxWidth, maxHeight, _numOfImages);

	// ---------- upload image data to GPU
	_allImages_cudaArrayWrapper->array3DCopy<unsigned char>(_imageDataBlock, cudaMemcpyHostToDevice);
	// attach to texture so that the kernel can access the data
	/*cudaChannelFormatDesc fmt;
	cudaGetChannelDesc(&fmt, _allImages_cudaArrayWrapper->_array3D);*/
	allImgsTexture.addressMode[0] = cudaAddressModeWrap; allImgsTexture.addressMode[1] = cudaAddressModeClamp; 
	allImgsTexture.filterMode = cudaFilterModeLinear;	allImgsTexture.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(allImgsTexture, _allImages_cudaArrayWrapper->_array3D));	// bind to texture	

	


	// testing
	testTextureArray();

	// prepare for the depth data
}

void PatchMatch::run()
{
	// ---------- initialize depthmap and SPM (selection probability map)
	//_depthMap = new Array2D_wrapper(maxWidth, maxHeight, _blockDim_x, _blockDim_y);
	//

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

inline __device__ void doTransform(float *col_prime, float *row_prime, float col, float row, int imageId)
{


}


inline __device__ float computeNCC(int imageId, float centerRow, float centerCol, float depth, int halfWindowSize)
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
			
			// 
			doTransform(&col_prime, &row_prime,  col + 0.5f, row + 0.5f, imageId);
			float Iprime = tex2DLayered(allImgsTexture, col_prime, row_prime, imageId);

			sum_I_I += (I * I);
			sum_Iprime_Iprime += (Iprime * Iprime);
			sum_I_Iprime += (Iprime * I);
			sum_I += I;
			sum_Iprime += Iprime;
		}
	}	
	float cost = (sum_I_I - 1/(2*halfWindowSize + 1) * sum_I * sum_I) * (sum_Iprime_Iprime - 1/(2*halfWindowSize + 1) * sum_Iprime * sum_Iprime );

	if(cost == 0)
		return -1;
	else
		return (sum_I_Iprime - 1/(2*halfWindowSize + 1) * sum_I * sum_Iprime )/cost;
}

inline __device__ int findMinCost(float *cost)
{
	int idx = cost[0] < cost[1]? 0:1;
	cost[1] = cost[idx];
	idx = cost[1] < cost[2]? 1:2;
	return idx;
}

__global__ void topToDown(int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples, curandState *randState, int randStatePitch, float depthRangeNear, float depthRangeFar, int halfWindowSize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	int threadId = threadIdx.x;

	if(col < refImageWidth)
	{
		__shared__ float depth_former_array[N];
		__shared__ float depth_current_array[N];
		__shared__ float sumOfSPMap[N];
		__shared__ float normalizedSPMap[N * TARGETIMGS];
		int s = (TARGETIMGS)>>7 + 1;
		float * depth_former = &depth_former_array[0];
		float * depth_current = &depth_current_array[0];
		__shared__ int selectedImages[ N * ( TARGETIMGS >>7) + N ];

		depth_former[threadId] = accessPitchMemory(depthMap, depthMapPitch, 0, col); 	
		for( int row = 1; row < refImageHeight; ++row)
		{
			
			depth_current[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col); 
			sumOfSPMap[threadId] = 0;
			
			if(threadId < N * s)
				selectedImages[threadId] = 0;
			__syncthreads();	
			//---------------------------------
			for(int i = 0; i<TARGETIMGS; i++)
				sumOfSPMap[threadId] += accessPitchMemory(SPMap,  SPMapPitch, row * TARGETIMGS + i, col);
			for(int i = 0; i<TARGETIMGS; i++)
				normalizedSPMap[threadId * i] = accessPitchMemory(SPMap,  SPMapPitch, row * TARGETIMGS + i, col)/ (sumOfSPMap[threadId] + FLT_MIN );	// devide by 0
			for(int i = 1; i<TARGETIMGS; i++)
				normalizedSPMap[threadId * i] += normalizedSPMap[threadId * (i-1)];
			
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
					if(randNum < normalizedSPMap[threadId * i])
					{
						int stateByte = selectedImages[(i>>7) * N + threadId];
						int pos = i - sizeof(int) * (i>>7);
						// check the bit first and set imageId
						bool state = CHECK_BIT(stateByte,pos);						
						if(!state)
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
					cost[0] =  computeNCC(imageId, (float)row, (float)col, depth_former[threadId], halfWindowSize);
					cost[1] =  computeNCC(imageId, (float)row, (float)col, depth_current[threadId], halfWindowSize);
					cost[2] =  computeNCC(imageId, (float)row, (float)col, randDepth, halfWindowSize);
				}
			}			

			// find the minimum cost id, and then put cost into global memory 
			int idx = findMinCost(cost);
			if(idx == 0)
				writePitchMemory(depthMap, depthMapPitch, row, col, depth_former[threadId] );
			else if(idx ==1)
				writePitchMemory(depthMap, depthMapPitch, row, col, depth_current[threadId] );
			else
				writePitchMemory(depthMap, depthMapPitch, row, col, randDepth );

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


PatchMatch::~PatchMatch()
{
	if(_imageDataBlock != NULL)
		delete []_imageDataBlock;
	if(_allImages_cudaArrayWrapper != NULL)
		delete _allImages_cudaArrayWrapper;
}

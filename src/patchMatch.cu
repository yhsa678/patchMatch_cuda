#include "patchMatch.h"

texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> allImgsTexture;


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

inline __device__ float accessPitchMemory(float *data, int pitch, int row, int col)
{
	return *((float*)((char*)data + pitch*row) + col);
}

//inline __device__ float accessSPMap(float *SPMap, int pitch, int row, int col, int imageId )
//{
//	float *address = (char *)SPMap + 
//
//}

__global__ void topToDown(int refImageWidth, int refImageHeight, float *depthMap, int depthMapPitch, float *SPMap, int SPMapPitch,
	int numOfSamples)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	int threadId = threadIdx.x;

	if(col < refImageWidth)
	{
		__shared__ float depth_former[N];
		__shared__ float depth_current[N];
		__shared__ float sumOfSPMap[N];
		__shared__ float normalizedSPMap[N * TARGETIMGS];
		__shared__ int selectedImages[ (N * TARGETIMGS)>>7 + 1 ];
		for( int row = 1; row < refImageHeight; ++row)
		{
			depth_former[threadId] = accessPitchMemory(depthMap, depthMapPitch, row - 1, col); 	
			depth_current[threadId] = accessPitchMemory(depthMap, depthMapPitch, row, col); 
			sumOfSPMap[threadId] = 0;
			int s = (N * TARGETIMGS)>>7 + 1;
			if(threadId < s)
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
			for(int j = 0; j < numOfSamples; j++)
			{
				float randNum = 0;
				for(int i = 0; i < TARGETIMGS; i++)
				{
					if(randNum < normalizedSPMap[threadId * i])
					{
						// set the bit

						// 
						break;
					}
				}
			}
			// check all the bits and do testing with 3 different depth: depth_former, depth_current, and a random depth



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

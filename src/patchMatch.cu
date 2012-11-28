#include "patchMatch.h"

texture<uchar, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> allImgsTexture;


PatchMatch::PatchMatch(const std::vector<Image> &allImage): _imageDataBlock(NULL), _allImages_cudaArrayWrapper(NULL)
{
	int numOfImages = allImage.size();
	if(numOfImages == 0)
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
	int sizeOfBlock = maxWidth * numOfChannels * maxHeight  * numOfImages;
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
	//for(int i = 0; i<6; i++)
	//	std::cout<< (unsigned int)(_imageDataBlock[maxWidth * numOfChannels * maxHeight  * numOfImages - i - 1]) << std::endl;

	// ---------- initialize array
	_allImages_cudaArrayWrapper = new CudaArray_wrapper(maxWidth, maxHeight, numOfImages);

	// ---------- upload data to GPU
	_allImages_cudaArrayWrapper->array3DCopy<unsigned char>(_imageDataBlock, cudaMemcpyHostToDevice);
	// attach to texture so that the kernel can access the data
	/*cudaChannelFormatDesc fmt;
	cudaGetChannelDesc(&fmt, _allImages_cudaArrayWrapper->_array3D);*/
	allImgsTexture.addressMode[0] = cudaAddressModeWrap; allImgsTexture.addressMode[1] = cudaAddressModeClamp; 
	allImgsTexture.filterMode = cudaFilterModeLinear;	allImgsTexture.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(allImgsTexture, _allImages_cudaArrayWrapper->_array3D));	// bind to texture	


	// testing
	//testTextureArray();

	// prepare for the depth data


}

__global__ void testTextureArray_kernel()
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	//printf("hello world. threadId: %d\n", id);
	//test it
	float fetchedData = 0;

	for(int j = 0; j < 1000; j++)
		for(int i = 0; i < 100; i++)
		{
			fetchedData += tex2DLayered( allImgsTexture, id + 0.5, i + 0.5,  0);
	
			//fetchedData += tex2DLayered( allImgsTexture, id + 0.5, i *2 + 0.5,  1);
	
			//fetchedData += tex2DLayered( allImgsTexture, id + 0.5, i *3 + 0.5,  2);
		}
	for(int j = 0; j < 1000; j++)
		for(int i = 0; i < 100; i++)
		{
			fetchedData += tex2DLayered( allImgsTexture, id + 0.5, i + 0.5,  1);		
		}
	for(int j = 0; j < 1000; j++)
		for(int i = 0; i < 100; i++)
		{
			fetchedData += tex2DLayered( allImgsTexture, id + 0.5, i + 0.5,  2);		
		}

	if( id == 0)
		printf("data: %f\n", fetchedData);
	
}

void PatchMatch::testTextureArray()
{
	dim3 blocksize(32,1,1);
	dim3 gridsize(1,1,1);
	/*CudaTimer t;
	t.startRecord();
	testTextureArray_kernel<<<gridsize, blocksize>>>();
	t.stopRecord();
	CudaCheckError();*/

}


PatchMatch::~PatchMatch()
{
	if(_imageDataBlock != NULL)
		delete []_imageDataBlock;
	if(_allImages_cudaArrayWrapper != NULL)
		delete []_allImages_cudaArrayWrapper;
}
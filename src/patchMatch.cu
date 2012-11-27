#include "patchMatch.h"


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
	_imageDataBlock = new char[sizeOfBlock]();
	// copy row by row
	//char *dest = _imageDataBlock;
	for(unsigned int i = 0; i < allImage.size(); i++)
	{	
		char *dest = _imageDataBlock + (maxWidth * numOfChannels * maxHeight) * i;
		char *source = (char* )(allImage[i]._imageData.data);

		for( int j = 0; j < maxHeight; j++)
		{
			if(j < allImage[i]._imageData.rows)
			{
				memcpy( (void *)dest, (void *)source, allImage[i]._imageData.step * sizeof(char));	

				dest += (maxWidth * numOfChannels);
				source += allImage[i]._imageData.step;
			}			
		}
	}
	for(int i = 0; i<6; i++)
		std::cout<< (unsigned int)(_imageDataBlock[maxWidth * numOfChannels * maxHeight  * numOfImages - i - 1]) << std::endl;

	// ---------- initialize array
	_allImages_cudaArrayWrapper = new CudaArray_wrapper(maxWidth, maxHeight, numOfImages);


}


PatchMatch::~PatchMatch()
{
	if(_imageDataBlock != NULL)
		delete []_imageDataBlock;
	if(_allImages_cudaArrayWrapper != NULL)
		delete []_allImages_cudaArrayWrapper;
}

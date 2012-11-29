#include "Image.h"
#include <string>
#include <iostream>
#include "patchMatch.h"

void main(int argc, char *argv[])
{
	std::string fileName = argv[1];
	float nearRange = 0;
	float farRange = 1;
	int halfWindowSize = 3;
	int blockDim_x = 32;
	int blockDim_y = 16;

	std::vector<Image> allImage;
	if(!readMiddleBurry(fileName, allImage))
	{
		std::cout<< "cannot read Image list file" << std::endl;
	}

	PatchMatch pm(allImage, nearRange, farRange, halfWindowSize, blockDim_x, blockDim_y);
	
}

#include "Image.h"
#include <string>
#include <iostream>
#include "patchMatch.h"

void main(int argc, char *argv[])
{
	float *x = new float[10]();
	std::cout << "sizeof(x) "<< sizeof(x) <<std::endl;
	std::cout << "sizeof(*x) "<< sizeof(*x) <<std::endl;


	std::string fileName = argv[1];
	float nearRange = 4.0;
	float farRange = 13.5;
	int refImageId = 0; // the reference id starts from 0
	int halfWindowSize = 5;
	int blockDim_x = 32;
	int blockDim_y = 16;
	int numOfSamples = 3;

	std::vector<Image> allImage;
	if(!readMiddleBurry(fileName, allImage))
	{
		std::cout<< "cannot read Image list file" << std::endl;
	}

	PatchMatch pm(allImage, nearRange, farRange, halfWindowSize, blockDim_x, blockDim_y, refImageId, numOfSamples);
	pm.run();
		
}

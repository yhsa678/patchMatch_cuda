#include "Image.h"
#include <string>
#include <iostream>
#include "patchMatch.h"

void main(int argc, char *argv[])
{
	std::string fileName = argv[1];

	std::vector<Image> allImage;
	if(!readMiddleBurry(fileName, allImage))
	{
		std::cout<< "cannot read Image list file" << std::endl;
	}

	

}

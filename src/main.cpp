#include "Image.h"
#include <string>
#include <iostream>
#include "patchMatch.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "utility_CUDA.h"


void main(int argc, char *argv[])
{
	//float x = 0.00000003001f;
	//float y = 0.0000123f;
	//float z = x * y;
	//std::cout<< "z: " << z << std::endl;



	if(argc <2)
	{
		std::cout << "Initialization file is mandatory to run the code" << std::endl;
		return;
	}
	setBestGPUDevice();

	std::string iniFileName = argv[1];

	std::string imageInfoFileName;
	float nearRange;
	float farRange;
	int refImageId; // the reference id starts from 0
	int halfWindowSize;
	int numOfSamples;
	float SPMAlpha;
	float gaussianSigma;
	int numOfIterations;

	try
	{
		boost::property_tree::ptree pt;
		boost::property_tree::ini_parser::read_ini(iniFileName.c_str(), pt);

		imageInfoFileName = pt.get<std::string>("params.imageInfoFileName");
		nearRange = pt.get<float>("params.depthRangeNear");
		farRange = pt.get<float>("params.depthRangeFar");
		refImageId = pt.get<int>("params.refImageId");
		halfWindowSize = pt.get<int>("params.halfWindowSize");
		numOfSamples = pt.get<int>("params.numOfSamples");

		SPMAlpha = pt.get<float>("params.SPMAlpha");
		gaussianSigma = pt.get<float>("params.gaussianSigma");
		numOfIterations = pt.get<float>("params.numOfIterations");
	}
	catch(std::exception const&  ex)
	{
		printf("Problems with config file: %s\n", ex.what());
		return;
	}

//-----------------------------------------------------------
	std::vector<Image> allImage;
	if(!readMiddleBurry(imageInfoFileName, allImage))
	{
		std::cout<< "cannot read Image list file" << std::endl;
	}

	int blockDim_x = 32;
	int blockDim_y = 16;
	PatchMatch pm(allImage, nearRange, farRange, halfWindowSize, blockDim_x, blockDim_y, refImageId, numOfSamples, SPMAlpha, gaussianSigma, numOfIterations );
	pm.runPatchMatch();

// save the file:
	pm._depthMap->saveToFile("depthMap.txt");		// save the depthmap

}

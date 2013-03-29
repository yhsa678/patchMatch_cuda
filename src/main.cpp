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
	

	std::string iniFileName = argv[1];

	std::string imageInfoFileName;
	float nearRange = -1.0f;
	float farRange = -1.0f;
	int refImageId; // the reference id starts from 0
	int halfWindowSize;
	int numOfSamples;
	float SPMAlpha;
	float gaussianSigma;
	int numOfIterations;
	int gpuId;
	std::string outputFileName;
	float orientationX;
	float orientationZ;

	try
	{
		boost::property_tree::ptree pt;
		boost::property_tree::ini_parser::read_ini(iniFileName.c_str(), pt);

		imageInfoFileName = pt.get<std::string>("params.imageInfoFileName");
		
		//nearRange = pt.get_optional<float>("params.depthRangeNear");
		//farRange = pt.get_optional<float>("params.depthRangeFar");
		boost::optional<float> nRange = pt.get_optional<float>("params.depthRangeNear");
		if(nRange)
			nearRange = nRange.get();

		boost::optional<float> fRange = pt.get_optional<float>("params.depthRangeFar");
		if(fRange)
			farRange = fRange.get();

		boost::optional<int> gpuIdOptional = pt.get_optional<int>("params.gpuId");
		if(gpuIdOptional)
			gpuId = gpuIdOptional.get();
		else
			gpuId = -1;

		boost::optional<float> orientationOptionalX = pt.get_optional<float>("params.orientationX");
		boost::optional<float> orientationOptionalZ = pt.get_optional<float>("params.orientationZ");
		if (orientationOptionalX && orientationOptionalZ)
		{
			orientationX = orientationOptionalX.get();
			orientationZ = orientationOptionalZ.get();
		}
		else
		{
			orientationX = 0.0f;
			orientationZ = 1.0f;
		}

		refImageId = pt.get<int>("params.refImageId");
		halfWindowSize = pt.get<int>("params.halfWindowSize");
		numOfSamples = pt.get<int>("params.numOfSamples");

		SPMAlpha = pt.get<float>("params.SPMAlpha");
		gaussianSigma = pt.get<float>("params.gaussianSigma");
		numOfIterations = pt.get<int>("params.numOfIterations");
		outputFileName = pt.get<std::string>("params.outputFileName"); 
	}
	catch(std::exception const&  ex)
	{
		printf("Problems with config file: %s\n", ex.what());
		return;
	}

	setBestGPUDevice(gpuId);

//-----------------------------------------------------------
	std::vector<Image> allImage;

	std::vector<std::pair<float, float> > depthRange;
	bool useNVM = false;
	if( readNVM(imageInfoFileName, allImage, depthRange ) )
	{
		std::cout << "using NVM files" << std::endl;
		useNVM = true;
	}
	else if(readMiddleBurry(imageInfoFileName, allImage))
	{
		std::cout << "using Middlebury files" << std::endl;
		useNVM = false;
	}
	else
	{
		std::cout << "cannot parse image files" << std::endl;
	}

	int blockDim_x = 32;
	int blockDim_y = 16;
	if(useNVM)
	{
		nearRange = depthRange[refImageId].first;
		farRange = depthRange[refImageId].second;
	}	

	PatchMatch pm(allImage, nearRange, farRange, halfWindowSize, blockDim_x, blockDim_y, refImageId, numOfSamples, SPMAlpha, gaussianSigma, numOfIterations , orientationX, orientationZ);
	pm.runPatchMatch();

// save the file:
	pm._depthMap->saveToFile(outputFileName.c_str());		// save the depthmap
}

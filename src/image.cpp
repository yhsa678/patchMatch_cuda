#include "Image.h"



void Image::updateCamParam(float *K, float *R, float *T, std::string imageFileName)
{
	_K = cv::Mat(3,3,CV_64F, K).clone();
	_R = cv::Mat(3,3,CV_64F, R).clone();
	_T = cv::Mat(3,1,CV_64F, T).clone();
	_C = -_R.t() * _T;

	_proj.create(3,4,CV_64F);
	for(int i = 0; i< 3; i++)
		// + 0 is necessary. See: http://opencv.willowgarage.com/documentation/cpp/core_basic_structures.html#Mat::row	
		_proj.col(i) = _R.col(i) + 0;	
	_proj.col(3) = _T + 0;
	_proj = _K * _proj;	

	_imageData = cv::imread(imageFileName);
}

bool readMiddleBurry(std::string fileName,  std::vector<Image> &allImages)
{
	std::ifstream in( fileName);
	if(!in.is_open())
	{
		return false;
	}
	unsigned int numOfImages; 
	in>>numOfImages;
	
	allImages.resize(numOfImages);
	 for(unsigned int i = 0; i < numOfImages; i++)
	 {
		std::string imageFileName;
		in>>imageFileName;
		float K[9], R[9], T[9];
		for(int j = 0; j < 9; j++)
			in>>K[j];		
		for(int j = 0; j < 9; j++)
			in>>R[j];
		for(int j = 0; j < 3; j++)
			in>>T[j];

		// update the K, R, T of im
		allImages[i].updateCamParam(K, R, T, imageFileName);
		
	 }
	
	return true;

}
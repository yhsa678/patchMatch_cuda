#include "Image.h"
#include "opencv/cv.h"

void Image::init_relative(const Image &refImg)
{
	_relative_R = _R * refImg._inverseR;
	_relative_T = _R * (_C - refImg._C);

	_H1 = _K * _relative_R * refImg._inverseK;
	cv::Mat normalVector = (cv::Mat_<float>(1,3) << 0, 0, 1);
	_H2 = _K * _relative_T * normalVector * refImg._inverseK;
}


void Image::updateCamParam(float *K, float *R, float *T, std::string imageFileName)
{
	_K = cv::Mat(3,3,CV_32F, K).clone();
	_R = cv::Mat(3,3,CV_32F, R).clone();
	_T = cv::Mat(3,1,CV_32F, T).clone();
	_C = -_R.t() * _T;

	_proj.create(3,4,CV_32F);
	for(int i = 0; i< 3; i++)
		// + 0 is necessary. See: http://opencv.willowgarage.com/documentation/cpp/core_basic_structures.html#Mat::row	
		_proj.col(i) = _R.col(i) + 0;	
	_proj.col(3) = _T + 0;
	_proj = _K * _proj;	
	cv::invert(_K, _inverseK); 
	cv::transpose(_R, _inverseR);

	_imageData = cv::imread(imageFileName);

	// convert color image to grey image
	if(_imageData.channels() != 1)
		cv::cvtColor(_imageData, _imageData, CV_BGR2GRAY);
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


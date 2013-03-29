
#ifndef IMAGE_H
#define IMAGE_H

#include <opencv/highgui.h>
#include <fstream>
#include <vector>
#include <string>

class Image{
	
	public:
		//	matrix for opencv	
		cv::Mat _R;
		cv::Mat _C;		// C = -R'T
		cv::Mat _T; 	// T = -RC
		cv::Mat _K;
		cv::Mat _proj;	// Proj = KR[I,-C] = k[R,T]

		cv::Mat _inverseK;
		cv::Mat _inverseR;
		cv::Mat _relative_R;
		cv::Mat _relative_T;
		cv::Mat _H1;
		cv::Mat _H2;

		void updateCamParam(float *K, float *R, float *T, std::string imageFileName);
		void updateCamParam(float *K, float *R, float *T, const cv::Mat &image);
		void init_relative(const Image &refImg, float orientationX, float orientationZ);

		cv::Mat _imageData;

};

bool readMiddleBurry(std::string fileName, std::vector<Image> &allImages);
bool readNVM(std::string fileName, std::vector<Image> &allImages, std::vector<std::pair<float, float>> depthRange);


#endif
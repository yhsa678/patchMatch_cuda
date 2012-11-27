




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
		void updateCamParam(float *K, float *R, float *T, std::string imageFileName);

		cv::Mat _imageData;

};

bool readMiddleBurry(std::string fileName, std::vector<Image> &allImages);


#endif
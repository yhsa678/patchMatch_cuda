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

void Image::updateCamParam(float *K, float *R, float *T, const cv::Mat &image)
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

	//_imageData = cv::imread(imageFileName);
	_imageData = image;

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

void quaternion2Rotation(double *R, double *q )
{
	double qq = sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
	double qw, qx, qy, qz;
	if(qq>0)
	{
		qw=q[0]/qq;
		qx=q[1]/qq;
		qy=q[2]/qq;
		qz=q[3]/qq;
	}else
	{
		qw = 1;
		qx = qy = qz = 0;
	}

	R[0]=(qw*qw + qx*qx- qz*qz- qy*qy );
	R[1]=(2*qx*qy -2*qz*qw );
	R[2]=(2*qy*qw + 2*qz*qx);
	R[3]=(2*qx*qy+ 2*qw*qz);
	R[4]=(qy*qy+ qw*qw - qz*qz- qx*qx);
	R[5]=(2*qz*qy- 2*qx*qw);
	R[6]=(2*qx*qz- 2*qy*qw);
	R[7]=(2*qy*qz + 2*qw*qx );
	R[8]=(qz*qz+ qw*qw- qy*qy- qx*qx);
}

void cameraCenter2Translation(double *R, double *c, double *T)
{	
	//T = -R*C;
	T[0] = -(R[0] * c[0] + R[1]* c[1] + R[2] * c[2]);
	T[1] = -(R[3] * c[0] + R[4]* c[1] + R[5] * c[2]);
	T[2] = -(R[6] * c[0] + R[7]* c[1] + R[8] * c[2]);
}

bool readNVM(std::string fileName, std::vector<Image> &allImages, std::vector<std::pair<float, float>> depthRange)
{
	std::ifstream in(fileName);

	int rotation_parameter_num = 4; 
	std::string token;
	if(in.peek() == 'N') 
	{
		in >> token; //file header
		if(!strstr(token.c_str(), "NVM_V3"))
		{
			in.close();
			return false;    //rotation as 3x3 matrix
		}
	}
	else
	{
		in.close();
		return false;
	}

	int ncam = 0, npoint = 0, nproj = 0;   
	// read # of cameras
	in >> ncam;  
	if(ncam <= 1) return false; 

	//read the camera parameters
	//camera_data.resize(ncam); // allocate the camera data
	allImages.resize(ncam);
	for(int i = 0; i < ncam; ++i)
	{
		std::string fileName;
		double f, q[9], c[3], d[2];
		double R[9], T[3];
		float K[9];
		in >> fileName >> f ;
		for(int j = 0; j < rotation_parameter_num; ++j) 
			in >> q[j]; 
		in >> c[0] >> c[1] >> c[2] >> d[0] >> d[1];

		quaternion2Rotation(R,q);
		cameraCenter2Translation(R, c, T);

		cv::Mat img = cv::imread(fileName.c_str());
		int width = img.cols;
		int height = img.rows;
		K[0] = static_cast<float>(f); K[1] = 0.0f;					K[2] = static_cast<float>(width)/2.0f;
		K[3] = 0.0f;				  K[4] = static_cast<float>(f); K[5] = height/2.0f;
		K[6] = 0.0f;				  K[7] = 0.0f;					K[8] = 1.0f;
		float R_float[9]; for(int j = 0; j< 9; j++) R_float[j] = static_cast<float>(R[j]);
		float T_float[3]; for(int j = 0; j< 3; j++) T_float[j] = static_cast<float>(T[j]);
		
		allImages[i].updateCamParam(K,R_float,T_float,img);
	}

	//////////////////////////////////////
	depthRange.resize(ncam);
	for(int i = 0; i < depthRange.size(); i++)
	{
		depthRange[i].first =   FLT_MAX;
		depthRange[i].second = -FLT_MAX;
	}

	in >> npoint;  
	if(npoint <= 0) 
	{
		in.close();
		return false; 
	}
	//read image projections and 3D points.

	//point_data.resize(npoint); 
	for(int i = 0; i < npoint; ++i)
	{
		float pt[3]; int cc[3], npj;
		in  >> pt[0] >> pt[1] >> pt[2]			// 3d points position
		>> cc[0] >> cc[1] >> cc[2] >> npj;		// 3d points color and number of measurements (npj). npj is used in the inside loop 
		for(int j = 0; j < npj; ++j)
		{
			int cidx, fidx; float imx, imy;
			in >> cidx >> fidx >> imx >> imy; //

			// compute the distance given the 3d position			
			cv::Mat points3D = (cv::Mat_<float>(4, 1) <<pt[0], pt[1], pt[2], 1.0f);
			cv::Mat points2D = allImages[cidx]._proj * points3D;
			float depth = points2D.at<float>(2,0);
			if(depth <= 0)		// sometimes changchang's code can generate wrong points
				continue;
			if(depth < depthRange[cidx].first)
				depthRange[cidx].first = depth;
			if(depth > depthRange[cidx].second)
				depthRange[cidx].second = depth;
		}
	}
	// loose the range a little bit:
	for(int i = 0; i<depthRange.size(); i++ )
	{
		float sizeOfRange = depthRange[i].second - depthRange[i].first;

		float ratio = 0.25f;
		float near = depthRange[i].first - ratio * sizeOfRange;
		while(	near <=0 ) // make
		{
			ratio /= 1.1f;
			near = depthRange[i].first - ratio * sizeOfRange;
		}
		depthRange[i].first = near;
		depthRange[i].second += 0.25f * sizeOfRange;
	}

	///////////////////////////////////////////////////////////////////////////////
	std::cout << ncam << " cameras; " << npoint << " 3D points; " << nproj << " projections\n";
	in.close();
	return true;

}
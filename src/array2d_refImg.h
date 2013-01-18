#ifndef ARRAY2D_REFIMG
#define ARRAY2D_REFIMG

#include "array2d_wrapper.h"
#include "cudaArray_wrapper.h"

class array2d_refImg
{
public:
	Array2D_wrapper<float> * _refImage_sum_II;
	Array2D_wrapper<float> * _refImage_sum_I;
	int _refWidth;
	int _refHeight;

	CudaArray_wrapper * _tempArray;

	array2d_refImg(int refWidth, int refHeight, int blockDim_x, int blockDim_y ):_refHeight(refHeight), _refWidth(refWidth), _refImage_sum_I(NULL), _refImage_sum_II(NULL), _tempArray(NULL)
	{
		_refImage_sum_II = new Array2D_wrapper<float>(refWidth, refHeight, blockDim_x, blockDim_y);
		_refImage_sum_I = new Array2D_wrapper<float>(refWidth, refHeight, blockDim_x, blockDim_y);

		_tempArray = new CudaArray_wrapper(refWidth, refHeight, 1);




	}
	void init(unsigned char *img, int halfWindowSize);
	

	~array2d_refImg()
	{
		if(_refImage_sum_I != NULL)
			delete _refImage_sum_I;
		if(_refImage_sum_II != NULL)
			delete _refImage_sum_II;
		if(_tempArray != NULL) 
			delete _tempArray;

	}
private:
	template<int FR> void FilterImage(cudaArray *dst, cudaArray *src);








	

};


#endif
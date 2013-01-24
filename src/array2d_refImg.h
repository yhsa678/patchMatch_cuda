#ifndef ARRAY2D_REFIMG
#define ARRAY2D_REFIMG

#include "array2d_wrapper.h"
#include "cudaArray_wrapper.h"

class Array2d_refImg
{
public:
	Array2D_wrapper<float> * _refImage_sum_II;
	Array2D_wrapper<float> * _refImage_sum_I;
	Array2D_wrapper<float> * _refImageData;

	int _refWidth;
	int _refHeight;

	CudaArray_wrapper * _tempArray;
	CudaArray_wrapper * _tempArray_float;

	Array2d_refImg(int refWidth, int refHeight, int blockDim_x, int blockDim_y, unsigned char* img ):_refHeight(refHeight), _refWidth(refWidth), _refImage_sum_I(NULL), _refImage_sum_II(NULL), _tempArray(NULL), _refImageData(NULL),
		_tempArray_float(NULL)
	{
		_refImage_sum_II = new Array2D_wrapper<float>(refWidth, refHeight, blockDim_x, blockDim_y);
		_refImage_sum_I = new Array2D_wrapper<float>(refWidth, refHeight, blockDim_x, blockDim_y);
		_refImageData = new Array2D_wrapper<float>(refWidth, refHeight, blockDim_x, blockDim_y);

		_tempArray = new CudaArray_wrapper(refWidth, refHeight, 1);
		_tempArray->array3DCopy<unsigned char>( img, cudaMemcpyHostToDevice);

		_tempArray_float = new CudaArray_wrapper(refWidth, refHeight, 1);
	}	Array2d_refImg(int refWidth, int refHeight, int blockDim_x, int blockDim_y ):_refHeight(refHeight), _refWidth(refWidth), _refImage_sum_I(NULL), _refImage_sum_II(NULL), _tempArray(NULL), _refImageData(NULL),
		_tempArray_float(NULL)
	{
		_refImage_sum_II = new Array2D_wrapper<float>(refWidth, refHeight, blockDim_x, blockDim_y);
		_refImage_sum_I = new Array2D_wrapper<float>(refWidth, refHeight, blockDim_x, blockDim_y);
		_refImageData = new Array2D_wrapper<float>(refWidth, refHeight, blockDim_x, blockDim_y);

		_tempArray = new CudaArray_wrapper(refWidth, refHeight, 1);

		_tempArray_float = new CudaArray_wrapper(refWidth, refHeight, 1);
	}

	~Array2d_refImg()
	{
		if(_refImage_sum_I != NULL)
			delete _refImage_sum_I;
		if(_refImage_sum_II != NULL)
			delete _refImage_sum_II;
		if(_refImageData != NULL)
			delete _refImageData;
		if(_tempArray != NULL) 
			delete _tempArray;
		if(_tempArray_float != NULL)
			delete _tempArray_float;
	}	void filterImage(int halfWindowSize);
private:
	template<int FR> void filterImage();

};


#endif
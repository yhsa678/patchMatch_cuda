#include "array2d_refImg.h"
#include "utility_CUDA.h"

#define THREADS_NUMBER_H 16
#define THREADS_NUMBER_V 12

texture<unsigned char, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> tex32F0;
//template<int FR>
__global__ void sumI_II_RowsKernel( float *output_I, float *output_sum_I, float *output_sum_II, int imageW, int imageH, int FR ) 
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

	if(ix >= imageW || iy >= imageH) return;
	
	float sum_I = 0.f;
	float sum_II = 0.f;

	float texValue;
	for(int k = -FR; k <= FR; k++)
	{ 		texValue = tex2DLayered(tex32F0, x + (float)k, y, 0); 
		//texValue = tex2D(tex32F0, x + (float)k, y);
		sum_I += texValue;
		sum_II += (texValue * texValue);
		//sum += tex2D(tex32F0, x + (float)k, y) * g_Kernel[FR - k]; 
	}

	//d_Dst[ IMAD(iy, imageW, ix) ] = sum;
	int outputAddr = iy * imageW + ix;
	output_sum_I[outputAddr] = sum_I;
	output_sum_II[outputAddr] = sum_II;

	output_I[outputAddr] = tex2DLayered(tex32F0, x, y, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Kernel Column convolution filter
////////////////////////////////////////////////////////////////////////////////
template<int FR> 
__global__ void sumI_II_ColsKernel( float *output, int imageW, int imageH )
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix >= imageW || iy >= imageH) return;

    float sum = 0.f;
	float texValue;
	for(int k = -FR; k <= FR; k++)
	{
		//texValue =  tex2D(tex32F0, x, y + (float)k);
		texValue = tex2DLayered(tex32F0, x,  y + (float)k, 0);
		sum += texValue;
		//sum += vtex2D(tex32F0, x, y + (float)k) * g_Kernel[FR - k];
	}
	int outputAddr = iy * imageW + ix;
	output[outputAddr] = sum;
	//d_Dst[IMAD(iy, imageW, ix)] = sum;
}


void Array2d_refImg::init(unsigned char *img)
{
	_tempArray->array3DCopy<unsigned char>( img, cudaMemcpyHostToDevice);
}

void Array2d_refImg::filterImage(int halfWindowSize)
{
	switch( halfWindowSize /*kernel radius*/ )
	{
		case 1:	 filterImage< 1>(); break;
		case 2:	 filterImage< 2>();	break;
		case 3:	 filterImage< 3>();	break;
		case 4:	 filterImage< 4>();	break;
		case 5:	 filterImage< 5>();	break;
		case 6:	 filterImage< 6>();	break;
		case 7:	 filterImage< 7>();	break;
		case 8:	 filterImage< 8>();	break;
		case 9:	 filterImage< 9>();	break;
		case 10: filterImage<10>();	break;
		case 11: filterImage<11>();	break;
		case 12: filterImage<12>();	break;
		case 13: filterImage<13>();	break;
		case 14: filterImage<14>();	break;
		case 15: filterImage<15>();	break;
		case 16: filterImage<16>();	break;
		case 17: filterImage<17>();	break;
		case 18: filterImage<18>();	break;
		case 19: filterImage<19>();	break;
		case 20: filterImage<20>();	break;
		case 21: filterImage<21>();	break;
		case 22: filterImage<22>();	break;
		default: std::cout<< "Warning: too large size of windows" << std::cout; exit(0);  break;
	}
}

template<int FR>
void Array2d_refImg::filterImage()
{
	dim3 threads(THREADS_NUMBER_H, THREADS_NUMBER_V);
	//dim3 blocks( iDivUp(m_nWidth, threads.x), iDivUp(m_nHeight, threads.y) ); //number of blocks required
	dim3 blocks( (_refWidth - 1)/threads.x + 1, (_refHeight - 1)/threads.y + 1);
	
	//horizontal pass:
		tex32F0.addressMode[0] = cudaAddressModeBorder; tex32F0.addressMode[1] = cudaAddressModeBorder; tex32F0.addressMode[2] = cudaAddressModeBorder;
	tex32F0.filterMode = cudaFilterModePoint; tex32F0.normalized = false;

	CUDA_SAFE_CALL(cudaBindTextureToArray(tex32F0, _tempArray->_array3D));
	sumI_II_RowsKernel<<<blocks, threads>>>(_refImageData->_array2D, _refImage_sum_I->_array2D, _refImage_sum_II->_array2D, _refWidth, _refHeight, FR);
//sumI_II_RowsKernel<FR><<<blocks, threads>>>(_refImage_sum_I->_array2D, _refImage_sum_II->_array2D, _refWidth, _refHeight);
	CudaCheckError();
	CUDA_SAFE_CALL(cudaUnbindTexture(tex32F0));

	//
	CUDA_SAFE_CALL(cudaMemcpy2DToArray( _tempArray->_array3D, 0, 0, _refImage_sum_I->_array2D, _refImage_sum_I->_pitchData, _refWidth*sizeof(float), _refHeight, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(tex32F0, _tempArray->_array3D));
	sumI_II_ColsKernel<FR><<<blocks, threads>>>(_refImage_sum_I->_array2D, _refWidth, _refHeight);
	CUDA_SAFE_CALL(cudaUnbindTexture(tex32F0));


	CUDA_SAFE_CALL(cudaMemcpy2DToArray( _tempArray->_array3D, 0, 0, _refImage_sum_II->_array2D, _refImage_sum_II->_pitchData, _refWidth*sizeof(float), _refHeight, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(tex32F0, _tempArray->_array3D));
	sumI_II_ColsKernel<FR><<<blocks, threads>>>(_refImage_sum_II->_array2D, _refWidth, _refHeight);
	CUDA_SAFE_CALL(cudaUnbindTexture(tex32F0));

}
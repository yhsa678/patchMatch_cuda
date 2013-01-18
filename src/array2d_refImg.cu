#include "array2d_refImg.h"
#include "utility_CUDA.h"

#define THREADS_NUMBER_H 16
#define THREADS_NUMBER_V 12
texture<float, cudaTextureType2D, cudaReadModeElementType> tex32F0;

template<int FR>
__global__ void sumI_II_RowsKernel( float *output_sum_I, float *output_sum_II, int imageW, int imageH ) 
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
	{ 
		texValue = tex2D(tex32F0, x + (float)k, y);
		sum_I += texValue;
		sum_II += (texValue * texValue);
		//sum += tex2D(tex32F0, x + (float)k, y) * g_Kernel[FR - k]; 
	}
    
	//d_Dst[ IMAD(iy, imageW, ix) ] = sum;
	int outputAddr = iy * imageW + ix;
	output_sum_I[outputAddr] = sum_I;
	output_sum_II[outputAddr] = sum_II;

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
		texValue =  tex2D(tex32F0, x, y + (float)k); 
		sum += texValue;
		//sum += vtex2D(tex32F0, x, y + (float)k) * g_Kernel[FR - k]; 
	}
	int outputAddr = iy * imageW + ix;
	output[outputAddr] = sum;
	//d_Dst[IMAD(iy, imageW, ix)] = sum;
}


void array2d_refImg::init(unsigned char *img, int halfWindowSize)
{
	_tempArray->array3DCopy<unsigned char>( img, cudaMemcpyHostToDevice);
	// bind texture

	cudaBindTextureToArray(tex32F0, _tempArray->_array3D);

	//



}


template<int FR> void array2d_refImg::FilterImage(cudaArray *dst, cudaArray *src)
{
	dim3 threads(THREADS_NUMBER_H, THREADS_NUMBER_V);
    //dim3 blocks( iDivUp(m_nWidth, threads.x), iDivUp(m_nHeight, threads.y) ); //number of blocks required
	dim3 blocks( (_refWidth - 1)/threads.x + 1, (_refHeight - 1)/threads.y + 1);
	
	//horizontal pass:
	CUDA_SAFE_CALL(cudaBindTextureToArray(tex32F0, _tempArray->_array3D));
	sumI_II_RowsKernel<FR><<<blocks, threads>>>(_refImage_sum_I->_array2D, _refImage_sum_II->_array2D, _refWidth, _refHeight);
	CUDA_SAFE_CALL(cudaUnbindTexture(tex32F0));

	//
	//cudaMemcpyToArray(m_cuaTmp, 0, 0, m_buf32FA, m_nWidth*m_nHeight*sizeof(float), cudaMemcpyDeviceToDevice);
	//cudaMemcpyToArray(_tempArray->_array3D, 0, 0, _refImage_sum_I->_array2D, _refImage
	CUDA_SAFE_CALL(cudaMemcpy2DToArray( _tempArray->_array3D, 0, 0, _refImage_sum_I->_array2D, _refImage_sum_I->_pitchData, _refWidth*sizeof(float), _refHeight, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(tex32F0, _tempArray->_array3D));
	sumI_II_ColsKernel<FR><<<blocks, threads>>>(_refImage_sum_I->_array2D, _refWidth, _refHeight);
	CUDA_SAFE_CALL(cudaUnbindTexture(tex32F0));


	CUDA_SAFE_CALL(cudaMemcpy2DToArray( _tempArray->_array3D, 0, 0, _refImage_sum_II->_array2D, _refImage_sum_II->_pitchData, _refWidth*sizeof(float), _refHeight, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(tex32F0, _tempArray->_array3D));
	sumI_II_ColsKernel<FR><<<blocks, threads>>>(_refImage_sum_II->_array2D, _refWidth, _refHeight);
	CUDA_SAFE_CALL(cudaUnbindTexture(tex32F0));
	

}
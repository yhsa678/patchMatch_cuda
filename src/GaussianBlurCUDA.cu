#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <cutil_inline.h>
#include "GaussianBlurCUDA.h"
#include "utility_CUDA.h"

//filter kernel width range (don't change these)
#define KERNEL_MAX_WIDTH 45       //do not change!!!
#define KERNEL_MIN_WIDTH  5       //do not change!!!
#define FILTER_WIDTH_FACTOR 5.0f

#define THREADS_NUMBER_H 16
#define THREADS_NUMBER_V 12

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel
//////////////////////////////////////////////////////////////////////////////// 
__device__ __constant__ float g_Kernel[KERNEL_MAX_WIDTH]; 

// declare texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex32F0;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
//Use unrolled innermost convolution loop
//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b){ return (a % b != 0) ? (a / b + 1) : (a / b); }
//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b){ return (a % b != 0) ?  (a - a % b + b) : a; }


////////////////////////////////////////////////////////////////////////////////
// Kernel Row convolution filter
////////////////////////////////////////////////////////////////////////////////
template<int FR> __global__ void convolutionRowsKernel( float *d_Dst, int imageW, int imageH )
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;
	if(ix >= imageW || iy >= imageH) return;
	
	float sum = 0.f;
   
	for(int k = -FR; k <= FR; k++){ sum += tex2D(tex32F0, x + (float)k, y) * g_Kernel[FR - k]; }
    
	d_Dst[ IMAD(iy, imageW, ix) ] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Kernel Column convolution filter
////////////////////////////////////////////////////////////////////////////////
template<int FR> __global__ void convolutionColsKernel( float *d_Dst, int imageW, int imageH )
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix >= imageW || iy >= imageH) return;

    float sum = 0.f;
	for(int k = -FR; k <= FR; k++){ sum += tex2D(tex32F0, x, y + (float)k) * g_Kernel[FR - k]; }
	
	d_Dst[IMAD(iy, imageW, ix)] = sum;
}


GaussianBlurCUDA::GaussianBlurCUDA(int width, int height, float sigma): m_nWidth(width), m_nHeight(height), m_paraSigma(sigma)
{
	cudaChannelFormatDesc floatTex  = cudaCreateChannelDesc<float>();
	
	//alloc cuda array:
	cudaMallocArray(&m_cuaSrc,  &floatTex, m_nWidth, m_nHeight);
	cudaMallocArray(&m_cuaTmp,  &floatTex, m_nWidth, m_nHeight);
	cudaMallocArray(&m_cuaBlur, &floatTex, m_nWidth, m_nHeight);
	
	//alloc system memory:
	cudaMalloc((void **)&m_buf32FA,  m_nWidth*m_nHeight*sizeof(float));
	
	//construct kernel for smoothing gradients
	float filter_kernel[KERNEL_MAX_WIDTH]; 
	CreateFilterKernel(m_paraSigma, filter_kernel, m_nKernelWidth);
	cudaMemcpyToSymbol(g_Kernel, filter_kernel, m_nKernelWidth*sizeof(float), 0, cudaMemcpyHostToDevice); //copy kernel to device memory.
}

GaussianBlurCUDA::~GaussianBlurCUDA()
{
	cudaFreeArray(m_cuaSrc);
	cudaFreeArray(m_cuaTmp); 
	cudaFreeArray(m_cuaBlur); 
	cudaFree(m_buf32FA); 
}

void GaussianBlurCUDA::CreateFilterKernel(float sigma, float* kernel, int& width)
{
	int i, sz;
	width = (int)(FILTER_WIDTH_FACTOR * sigma);
	if( width%2 == 0 ){ width+=1; }
	sz = (width-1)>>1;

	if(width > KERNEL_MAX_WIDTH)
	{
		//filter size truncation
		sz = KERNEL_MAX_WIDTH >> 1;
		width = KERNEL_MAX_WIDTH;
	}else if(width < KERNEL_MIN_WIDTH)
	{
		sz = KERNEL_MIN_WIDTH >> 1;
		width = KERNEL_MIN_WIDTH;
	}

	float rv = -0.5f/(sigma*sigma), v, ksum = 0.f; 

	// pre-compute filter
	for( i = -sz ; i <= sz ; ++i) {
		kernel[i+sz] = v = exp( i * i * rv ) ;
		ksum += v;
	}

	//normalize the kernel
	rv = 1.0f/ksum; for(i=0; i<width; i++) kernel[i]*=rv;

	//for(i = 0; i < width; i++)
	//	kernel[i] = 1.0f/(float)width/(float)width;
}

template<int FR> void GaussianBlurCUDA::FilterImage(cudaArray *dst, cudaArray *src)
{
	dim3 threads(THREADS_NUMBER_H, THREADS_NUMBER_V);
    dim3 blocks( iDivUp(m_nWidth, threads.x), iDivUp(m_nHeight, threads.y) ); //number of blocks required
	
	//horizontal pass:
	cudaBindTextureToArray(tex32F0, src);
	convolutionRowsKernel<FR><<<blocks, threads>>>( m_buf32FA, m_nWidth, m_nHeight );
	cudaUnbindTexture(tex32F0);
	cudaMemcpyToArray(m_cuaTmp, 0, 0, m_buf32FA, m_nWidth*m_nHeight*sizeof(float), cudaMemcpyDeviceToDevice);
	
	//vertical pass:
	cudaBindTextureToArray(tex32F0, m_cuaTmp);
	convolutionColsKernel<FR><<<blocks, threads>>>( m_buf32FA, m_nWidth, m_nHeight );
	cudaUnbindTexture(tex32F0);
	cudaMemcpyToArray(    dst, 0, 0, m_buf32FA, m_nWidth*m_nHeight*sizeof(float), cudaMemcpyDeviceToDevice);	
}


void GaussianBlurCUDA::Filter(cudaArray *dst, cudaArray *src)
{
	switch( m_nKernelWidth>>1 /*kernel radius*/ )
	{
		case 2:	 FilterImage< 2>(dst, src);	break;
		case 3:	 FilterImage< 3>(dst, src);	break;
		case 4:	 FilterImage< 4>(dst, src);	break;
		case 5:	 FilterImage< 5>(dst, src);	break;
		case 6:	 FilterImage< 6>(dst, src);	break;
		case 7:	 FilterImage< 7>(dst, src);	break;
		case 8:	 FilterImage< 8>(dst, src);	break;
		case 9:	 FilterImage< 9>(dst, src);	break;
		case 10: FilterImage<10>(dst, src);	break;
		case 11: FilterImage<11>(dst, src);	break;
		case 12: FilterImage<12>(dst, src);	break;
		case 13: FilterImage<13>(dst, src);	break;
		case 14: FilterImage<14>(dst, src);	break;
		case 15: FilterImage<15>(dst, src);	break;
		case 16: FilterImage<16>(dst, src);	break;
		case 17: FilterImage<17>(dst, src);	break;
		case 18: FilterImage<18>(dst, src);	break;
		case 19: FilterImage<19>(dst, src);	break;
		case 20: FilterImage<20>(dst, src);	break;
		case 21: FilterImage<21>(dst, src);	break;
		case 22: FilterImage<22>(dst, src);	break;
		default: break;
	}
}

int GaussianBlurCUDA::Filter( float* dst, const float* src )
{
	cudaMemcpyToArray(m_cuaSrc, 0, 0, src, m_nWidth * m_nHeight * sizeof(float), cudaMemcpyHostToDevice);
	Filter(m_cuaBlur, m_cuaSrc);
	cudaMemcpy(dst, m_buf32FA, m_nWidth * m_nHeight * sizeof(float), cudaMemcpyDeviceToHost); //GPU memory to CPU memory copy - slow!!!
	return 0;
}


int GaussianBlurCUDA::FilterMultipleImages(float *data, int pitch, int depth)
{
	for(int i = 0; i < depth; i++)
	{
		CUDA_SAFE_CALL(cudaMemcpy2DToArray(m_cuaSrc, 0, 0, data + i * pitch/sizeof(float)  * m_nHeight, pitch, m_nWidth * sizeof(float), m_nHeight, cudaMemcpyDeviceToDevice));
		Filter(m_cuaBlur, m_cuaSrc);
		CudaCheckError();
		CUDA_SAFE_CALL(cudaMemcpy2DFromArray(data + i * pitch/sizeof(float) * m_nHeight, pitch, m_cuaBlur, 0, 0, m_nWidth * sizeof(float), m_nHeight, cudaMemcpyDeviceToDevice)); 
		//CUDA_SAFE_CALL(cudaMemcpy2DFromArray(data + i * pitch/sizeof(float) * m_nHeight, pitch, m_cuaSrc, 0, 0, m_nWidth * sizeof(float), m_nHeight, cudaMemcpyDeviceToDevice)); 
	}
	return 0;
}
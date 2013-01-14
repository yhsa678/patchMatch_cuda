#pragma once

#include <cuda_runtime.h>

class GaussianBlurCUDA
{
public:
	GaussianBlurCUDA(int width, int height, float sigma); //the filter size will be filterwidthfactor*sigma*2+1
	~GaussianBlurCUDA();
	int Filter( float* dst, const float* src );
	void Filter(cudaArray *dst, cudaArray *src);
private:
	template<int FR> void FilterImage(cudaArray *dst, cudaArray *src); //filter width
	void CreateFilterKernel(float sigma, float* kernel, int& width);
private:
	int m_nWidth, m_nHeight, m_nKernelWidth;
	float m_paraSigma;
private:
	//cuda array
	cudaArray* m_cuaSrc; //store the src image
	cudaArray* m_cuaTmp;
	cudaArray* m_cuaBlur;
	float*	   m_buf32FA;
};
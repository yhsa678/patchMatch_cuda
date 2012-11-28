#ifndef UTILITY_CUDA_H
#define UTILITY_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <string>

#define CUDA_SAFE_CALL(err) _CUDA_SAFE_CALL( err,__FILE__, __LINE__)
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

void _CUDA_SAFE_CALL( cudaError_t err, std::string file = __FILE__, int line = __LINE__);
void __cudaCheckError( const char *file, const int line );



class CudaTimer{
private:
	cudaEvent_t start;
	cudaEvent_t	stop;
	float elapsedTime;
public:
	CudaTimer()
	{
		CUDA_SAFE_CALL( cudaEventCreate( &start));
		CUDA_SAFE_CALL( cudaEventCreate( &stop));
	}
	~CudaTimer()
	{
		CUDA_SAFE_CALL( cudaEventDestroy(start));
		CUDA_SAFE_CALL( cudaEventDestroy(stop));
	}
	void startRecord()
	{
		CUDA_SAFE_CALL( cudaEventRecord( start, 0 ));
	}
	void stopRecord()
	{
		CUDA_SAFE_CALL( cudaEventRecord( stop, 0));
		CUDA_SAFE_CALL( cudaEventSynchronize( stop));
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime, start, stop));
		printf("it takes: %f\n" , elapsedTime);
	}
};


#endif
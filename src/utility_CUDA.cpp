#include "utility_CUDA.h"
#include <vector>

void _CUDA_SAFE_CALL( cudaError_t err, std::string file, int line)
{
#ifdef _DEBUG
	if (err != cudaSuccess) {
		//std::cout<< cudaGetErrorString( err ) << " in file: " << file << " at line: " << line << std::endl;
		printf( "%s in %s at line %i\n", cudaGetErrorString( err ),
			file.c_str(), line );
		exit( EXIT_FAILURE );
	}
#endif
}

void __cudaCheckError( const char *file, const int line )
{
#ifdef _DEBUG
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
	return;
#endif
}


void showGreyImage(unsigned char *data, int width, int height)
{
//#ifndef _WIN64
//	imdebug("lum b=8 w=%d h=%d %p", width, height, data);
//#endif
}

void showRGBImage(unsigned char *data, int width, int height)
{
//#ifndef _WIN64
//	imdebug("rgb w=%d h=%d %p", width, height, data);
//#endif

}

void viewData1DDevicePointer(float * data, int size)
{
	thrust::device_ptr<float> dev_ptr(data);
	thrust::copy(dev_ptr, dev_ptr + size, std::ostream_iterator<float>(std::cout, " "));
	std::cout<< std::endl;
}
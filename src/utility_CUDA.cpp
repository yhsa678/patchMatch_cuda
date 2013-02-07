#include "utility_CUDA.h"
#include <vector>

void _CUDA_SAFE_CALL( cudaError_t err, std::string file, int line)
{
//#ifdef _DEBUG
	if (err != cudaSuccess) {
		//std::cout<< cudaGetErrorString( err ) << " in file: " << file << " at line: " << line << std::endl;
		printf( "%s in %s at line %i\n", cudaGetErrorString( err ),
			file.c_str(), line );
		exit( EXIT_FAILURE );
	}
//#endif
}

void __cudaCheckError( const char *file, const int line )
{
//#ifdef _DEBUG
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
//#endif
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

namespace{
	bool compareDevice(const cudaDeviceProp &d1, const cudaDeviceProp &d2)
	{
		bool result = (d1.major > d2.major) || 
			((d1.major == d2.major) && (d1.minor > d2.minor)) ||
			((d1.major == d2.major) && (d1.minor == d2.minor) && (d1.multiProcessorCount > d2.multiProcessorCount));
		return result;
	}
}

void setBestGPUDevice()
{
	int number_of_devices;
	cudaGetDeviceCount(&number_of_devices);
	if (number_of_devices > 1) {
		cudaDeviceProp *allDevice = new cudaDeviceProp[number_of_devices];

		for (int device_index = 0; device_index < number_of_devices; device_index++) {
			cudaGetDeviceProperties(&allDevice[device_index], device_index);
		}
		std::sort(allDevice, allDevice+number_of_devices, compareDevice);
		int best_gpu = 0;
		CUDA_SAFE_CALL(cudaChooseDevice(&best_gpu, &allDevice[0]));
		cudaSetDevice(best_gpu);
		delete []allDevice;
	}
	else if(number_of_devices == 0)
	{
		printf("No Nvidia GPU is detected in the machine");
		exit(EXIT_FAILURE);
	}
	else
	{		
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, 0);
		std::cout<< "the device name is: " << device.name << std::endl;
		cudaSetDevice(0);
	}
}

#include "utility_CUDA.h"

void _CUDA_SAFE_CALL( cudaError_t err, std::string file, int line)
{
	if (err != cudaSuccess) {
		//std::cout<< cudaGetErrorString( err ) << " in file: " << file << " at line: " << line << std::endl;
		printf( "%s in %s at line %i\n", cudaGetErrorString( err ),
			file.c_str(), line );
		exit( EXIT_FAILURE );
	}
}
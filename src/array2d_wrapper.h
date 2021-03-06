#ifndef ARRAY2D_WRAPPER
#define ARRAY2D_WRAPPER
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "utility_CUDA.h"

#include <string>
#include <fstream>
#include <iterator>

__global__ void generate_kernel_float( curandState *state, int statePitch,  float * result, int resultPitch, int width, int height, float rangeStart, float rangeEnd );
__global__ void generate_kernel_float_withDepth( curandState *state, int statePitch,  float * result, int resultPitch, int width, int height, int depth, float rangeStart, float rangeEnd );

template<class T>
class Array2D_wrapper{
public:
	T *_array2D;
	int _pitchData;
	Array2D_wrapper(int width, int height, int blockDim_x, int blockDim_y, int depth = 1):
			_width(width), _height(height), _blockDim_x(blockDim_x), _blockDim_y(blockDim_y),
				_depth(depth), _array2D(NULL)
	{
		size_t pitchData;
		CUDA_SAFE_CALL(cudaMallocPitch((void**)&_array2D, &pitchData, static_cast<size_t>(_width * sizeof(T)), static_cast<size_t>(_height * _depth)));
		_pitchData = static_cast<int>(pitchData);
		
		// compute grid and block size
		computeCUDAConfig();
	}

	~Array2D_wrapper()
	{
		if(_array2D != NULL)
			CUDA_SAFE_CALL(cudaFree((void *) _array2D));
	}
	void randNumGen(float rangeStart, float rangeEnd, curandState * devStates, int pitchState);
	int getWidth();
	int getHeight();
	int getDepth();

	void copyData(T* ptr, int pitch, enum cudaMemcpyKind kind)
	{
		// cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost
		if(kind == cudaMemcpyDeviceToHost)
			CUDA_SAFE_CALL(cudaMemcpy2D((void *)ptr, pitch, (void *)_array2D, (size_t)_pitchData, _width * sizeof(T),  _height*_depth, 	kind));
		else
			CUDA_SAFE_CALL(cudaMemcpy2D( (void *)_array2D, (size_t)_pitchData, (void *)ptr, pitch, _width * sizeof(T) ,_height*_depth, 	kind));

	}

	void saveToFile(std::string fileName)
	{
		std::fstream fout(fileName, 'w');
		fout<<_width<<"&"<<_height<< "&"<<_depth<< "&";
		fout.close();

		std::fstream foutBinary(fileName, std::ios_base::out|std::ios_base::binary|std::ios_base::app);		
		T *destPtr = new T[_width * _height * _depth];
		/*destPtr[0] = static_cast<T>(_width);
		destPtr[1] = static_cast<T>(_height);
		destPtr[2] = static_cast<T>(_depth);*/
		copyData(destPtr, _width * sizeof(T) , cudaMemcpyDeviceToHost);
		std::copy(destPtr, destPtr + _width * _height * _depth, std::ostream_iterator<T>(fout, " "));
		/*fout.write((char*)&_width, sizeof(int));
		fout.write((char*)&_height, sizeof(int));
		fout.write((char*)&_depth, sizeof(int));*/
		foutBinary.write((char*)destPtr, sizeof(T) * _width*_height*_depth);
	
		delete []destPtr;
		foutBinary.close();



		 //std::ofstream out(filename,std::ios_base::out|std::ios_base::binary);
   //   verify(out.is_open(),"Failed to open iamge data file.");
   //   // Write image stat.
   //   ImageFileStat stat;
   //   stat.width = image.width();
   //   stat.height = image.height();
   //   stat.numChannels = image.numChannels();
   //   stat.bitDepth = sizeof(Elem)*8;
   //   // TODO: Create a mechanism for getting a type constant.
   //   //       Then type can be stored exactly.
   //   unsigned int type = 0xffffffff;
   //   out.write((char*)&stat.width,sizeof(unsigned int));
   //   out.write((char*)&stat.height,sizeof(unsigned int));
   //   out.write((char*)&stat.numChannels,sizeof(unsigned int));
   //   out.write((char*)&stat.bitDepth,sizeof(unsigned int));
   //   out.write((char*)&type,sizeof(unsigned int));
   //   // Write image data.
   //   out.write((char*)&image(0,0,0),sizeof(Elem)*image.width()*image.height()*image.numChannels());
	}

	void saveToFile(std::string fileName, int layer)
	{
		std::fstream fout(fileName, 'w');
		T * destPtr = new T[_width * _height];
		CUDA_SAFE_CALL(cudaMemcpy2D((void *)destPtr, _width * sizeof(T), (void *)(_array2D + layer * _height * _pitchData/sizeof(T)), _pitchData, 	_width * sizeof(T),  _height, 	cudaMemcpyDeviceToHost));
		std::copy(destPtr, destPtr + _width * _height, std::ostream_iterator<T>(fout, "\n"));
		delete []destPtr;
		fout.close();
	}

protected:
	void computeCUDAConfig();

	//size_t _width;
	int _height;
	int _depth;
	int _width;

	dim3 _blockSize;
	int _blockDim_x;
	int _blockDim_y;
	dim3 _gridSize;

};

template<class T>
int Array2D_wrapper<T> :: getHeight()
{
	return _height;
}

template<class T>
int Array2D_wrapper<T> :: getWidth()
{
	return _width;
}

template<class T>
int Array2D_wrapper<T> :: getDepth()
{
	return _depth;
}

template<class T>
void Array2D_wrapper<T> :: randNumGen(float rangeStart, float rangeEnd, curandState * devStates, int pitchState)
{		
	if(_depth == 1)
		generate_kernel_float <<<_gridSize, _blockSize >>>(devStates, pitchState, _array2D, _pitchData, _width, _height, rangeStart, rangeEnd );	
	else
		generate_kernel_float_withDepth<<<_gridSize, _blockSize >>>(devStates, pitchState, _array2D, _pitchData, _width, _height, _depth, rangeStart, rangeEnd );	
	CudaCheckError();
}

template<class T>
void Array2D_wrapper<T>::computeCUDAConfig()
{
	_blockSize.x = _blockDim_x;
	_blockSize.y = _blockDim_y;
	_blockSize.z = 1;

	_gridSize.x = (_width - 1)/ static_cast<int>(_blockDim_x) + 1 ;
	_gridSize.y = (_height - 1)/ static_cast<int>(_blockDim_y) + 1 ;
	_gridSize.z = 1;
}

#endif
#include "cudaTranspose.h"

#define TILE_DIM 32 // do not change this
#define BLOCK_ROWS 8 // TILE_DIM should be dividable by BLOCK_ROWS. Do not change it unless you know what happens

//--------------------------------------------------------------------
__global__ void transpose2DData(float *odata, float *idata, int width, int height, int pitchIdata, int pitchOdata)
{
	int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
	
	//pitchIdata /= sizeof(int);
	//pitchOdata /= sizeof(int);

	int index_in = xIndex + (yIndex) * pitchIdata; // size of int is 4
	//int index_in;
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
	int yInd;
	int xInd;
	int idx;
	int idy;
	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) 
	{

		//if(yIndex + i < height && xIndex < width)
		//	tile[threadIdx.y+i][threadIdx.x] = idata[index_in + i * pitchIdata];		//	correct one, but slower. I do not know why. related to __syncthreads()?
		
		yInd = min( yIndex, height - i - 1); 		
		xInd = min( xIndex, width - 1);
		index_in = xInd + (yInd) * pitchIdata;
		idx = min(threadIdx.x, width - 1 - blockIdx.x * TILE_DIM);
		idy = min(threadIdx.y, height -1 - blockIdx.y * TILE_DIM);		
		tile[idy + i][idx] = idata[index_in + i * pitchIdata];		
	}
	__syncthreads();
	// ------------------------------------------------------------------------
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex) * pitchOdata;		
	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
		//if( (blockIdx.y * TILE_DIM + threadIdx.x < height) && (blockIdx.x * TILE_DIM + threadIdx.y + i< width) )		correct one
		if( xIndex < height && yIndex + i< width )				// here is a little bit tricky
			odata[index_out + i * pitchOdata] = tile[threadIdx.x][threadIdx.y+i];
	}
}

void cudaTranspose::transpose2dData(float * input, float *output, int width, int height, int pitchInput, int pitchOutput)
{
	dim3 blockDim(TILE_DIM, BLOCK_ROWS, 1);
	dim3 gridDim;
	gridDim.x = (width - 1)/TILE_DIM + 1;
	gridDim.y = (height - 1)/TILE_DIM + 1;
	pitchInput /= sizeof(float);
	pitchOutput /= sizeof(float);

	transpose2DData<<< gridDim, blockDim>>>(output, input, width, height, pitchInput, pitchOutput);
}
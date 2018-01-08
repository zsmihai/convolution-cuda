
#include <assert.h>
#include <cooperative_groups.h>
#include "types.h"
#include <vector_types.h>
#include <device_functions.h>

#define BLOCKDIM_H 16
#define BLOCKDIM_W 16


namespace cg = cooperative_groups;


__global__ void ConvolutionKernel(
	BYTE * DestinationMatrix,
	BYTE * SourceMatrix,
	int ImageWidth,
	int ImageHeight,
	BYTE * KernelMatrix,
	int KernelRadius
)
{
	cg::thread_block cta = cg::this_thread_block();
	// matrice cu width BLOCKDIM_X + 2 * kernelRadius si heigth BLOCKDIM_Y + 2 * kernelRadius
	extern __shared__ BYTE blockMatrix[];

	int blockIndex;
	int sourceIndexX, sourceIndexY, sourceIndex;
	int blockWidth = BLOCKDIM_W + 2 * KernelRadius;
	int blockHeight = BLOCKDIM_H + 2 * KernelRadius;
	int x, y;

	//load pixel
	blockIndex = threadIdx.y * blockWidth + threadIdx.x;
	sourceIndexX = blockIdx.x * BLOCKDIM_W + threadIdx.x - KernelRadius;
	sourceIndexY = blockIdx.y * BLOCKDIM_H + threadIdx.y - KernelRadius;
	sourceIndex = sourceIndexY * ImageWidth + sourceIndexX;
	blockMatrix[blockIndex] = (sourceIndexX >= 0 && sourceIndexX < ImageWidth && sourceIndexY >= 0 && sourceIndexY < ImageHeight) ? SourceMatrix[sourceIndex] : 0;

	__syncthreads();
	
	if (threadIdx.x < BLOCKDIM_W + KernelRadius && threadIdx.x >= KernelRadius &&
		threadIdx.y < BLOCKDIM_H + KernelRadius && threadIdx.y >= KernelRadius)
	{
		BYTE accumulator = 0;

		for (int kernelX = -KernelRadius; kernelX <= KernelRadius; kernelX++)
		{

			for (int kernelY = -KernelRadius; kernelY <= KernelRadius; kernelY++)
			{
				int kernelIndex = (kernelY + KernelRadius) * (2 * KernelRadius + 1) + (kernelX + KernelRadius);
				int blockMatrixIndex = (threadIdx.y + kernelY) * blockWidth + (threadIdx.x + kernelX);
				accumulator += (KernelMatrix[kernelIndex] * blockMatrix[blockMatrixIndex]);
			}
		}

		DestinationMatrix[sourceIndex] = accumulator;
	}
	
}

extern "C"
void
ConvolutionGPU(
	BYTE * DestinationMatrix,
	BYTE * SourceMatrix,
	int ImageWidth,
	int ImageHeight,
	BYTE * KernelMatrix,
	int KernelRadius
)
{
	assert(ImageWidth % BLOCKDIM_W == 0);
	assert(ImageHeight % BLOCKDIM_H == 0);

	dim3 blocks(ImageWidth / BLOCKDIM_W, ImageHeight / (BLOCKDIM_H));
	dim3 threads(BLOCKDIM_W + 2 * KernelRadius, BLOCKDIM_H + 2 * KernelRadius);


	ConvolutionKernel <<<blocks, threads, (BLOCKDIM_W + 2 * KernelRadius) * (BLOCKDIM_H + 2 * KernelRadius)>>> (
		DestinationMatrix,
		SourceMatrix,
		ImageWidth,
		ImageHeight,
		KernelMatrix,
		KernelRadius
	);
}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "types.h"
#include "fileUtils.h"

cudaError_t
convolutionWithCuda(
	PBYTE *	DestinationMatrix,
	BYTE *	SourceMatrix,
	int ImageWidth,
	int ImageHeight,
	BYTE * KernelMatrix,
	int KernelRadius
);

extern "C"
void
ConvolutionGPU(
	BYTE * DestinationMatrix,
	BYTE * SourceMatrix,
	int ImageWidth,
	int ImageHeight,
	BYTE * KernelMatrix,
	int KernelRadius
);

int main()
{
    PBYTE matrix;
	PBYTE kernel;
	unsigned int matrixWidth, matrixHeight;
	unsigned int kernelRadius, kernelLength;
	PBYTE resultMatrix;

	if (!ReadSampleMatrix(&matrix, &matrixWidth, &matrixHeight))
	{
		return 1;
	}

	if (!ReadKernel(&kernel, &kernelLength, NULL))
	{
		return 1;
	}
	
	kernelRadius = (kernelLength - 1) / 2;

	printf("Matrices read\n");

    cudaError_t cudaStatus = convolutionWithCuda(&resultMatrix, matrix, matrixWidth, matrixHeight, kernel, kernelRadius);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "convolutionWithCuda failed!");
        return 1;
    }

    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	
    return 0;
}

cudaError_t
convolutionWithCuda(
	PBYTE *	DestinationMatrix,
	BYTE *	SourceMatrix,
	int ImageWidth,
	int ImageHeight,
	BYTE * KernelMatrix,
	int KernelRadius
)
{
	BYTE *deviceSourceMatrix = NULL;
    BYTE *deviceDestinationMatrix = NULL;
	BYTE *resultMatrix = NULL;	
    BYTE *kernel = NULL;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%d\n", cudaStatus);
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	resultMatrix = (PBYTE)malloc(ImageWidth * ImageHeight * sizeof(BYTE));
	
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&deviceSourceMatrix, ImageHeight * ImageWidth * sizeof(BYTE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&deviceDestinationMatrix, ImageHeight * ImageWidth * sizeof(BYTE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&kernel, (KernelRadius * 2 + 1)* (KernelRadius * 2 + 1)* sizeof(BYTE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(deviceSourceMatrix, SourceMatrix, ImageHeight * ImageWidth * sizeof(BYTE), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(kernel, KernelMatrix, (KernelRadius * 2 + 1)* (KernelRadius * 2 + 1) * sizeof(BYTE), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	ConvolutionGPU(
		deviceDestinationMatrix, deviceSourceMatrix, ImageWidth, ImageHeight, kernel, KernelRadius
	);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(resultMatrix, deviceDestinationMatrix, ImageHeight * ImageWidth * sizeof(BYTE), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(deviceDestinationMatrix);
    cudaFree(deviceSourceMatrix);
    cudaFree(kernel);
    
	*DestinationMatrix = resultMatrix;

    return cudaStatus;
}

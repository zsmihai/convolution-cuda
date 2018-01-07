#include "fileUtils.h"


bool
ReadSampleMatrix(
	PBYTE *	Matrix,
	unsigned int * MatrixWidth,
	unsigned int * MatrixHeight
)
{
	return ReadMatrix("SampleMatrix.txt", Matrix, MatrixWidth, MatrixHeight);
}

bool
readKernel(
	PBYTE *	Matrix,
	unsigned int * MatrixWidth,
	unsigned int * MatrixHeight
)
{
	return ReadMatrix("SampleKernel.txt", Matrix, MatrixWidth, MatrixHeight);
}

bool
ReadMatrix(
	const char* Filename,
	PBYTE *	Matrix,
	unsigned int * MatrixWidth,
	unsigned int * MatrixHeight
)
{
	FILE* matrixFile;
	PBYTE matrix;
	unsigned int matrixWidth;
	unsigned int matrixHeight;
	unsigned int rowIndex, columnIndex;

	matrixFile = fopen(Filename, "r");
	if (NULL == matrixFile)
	{
		//file opening failed
		fprintf(stderr, "ReadMatrix: fopen failed with error code %d.\n", errno);
		return false;
	}

	fscanf(matrixFile, "%d %d", &matrixHeight, &matrixWidth);

	matrix = (PBYTE)malloc(matrixWidth * matrixHeight * sizeof(BYTE));

	for (rowIndex = 0; rowIndex < matrixHeight; rowIndex++)
	{
		for (columnIndex = 0; columnIndex < matrixWidth; columnIndex++)
		{
			fscanf(matrixFile, "%d", &matrix[rowIndex * matrixWidth + columnIndex]);
		}
	}

	*Matrix = matrix;
	*MatrixHeight = matrixHeight;
	*MatrixWidth = matrixWidth;

	return true;
}
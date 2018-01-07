#include "types.h"
#include <stdio.h>
#include <malloc.h>
#include <errno.h>

bool
ReadMatrix(
	const char* Filename,
	PBYTE *	Matrix,
	unsigned int * MatrixWidth,
	unsigned int * MatrixHeight
);

bool
ReadSampleMatrix(
	PBYTE *	Matrix,
	unsigned int * MatrixWidth,
	unsigned int * MatrixHeight
);

inline
void
FreeMatrix(
	PBYTE * Matrix
)
{
	free(*Matrix);
	*Matrix = NULL;
}

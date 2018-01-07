#include "types.h"
#include <stdio.h>
#include <malloc.h>
#include <errno.h>

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

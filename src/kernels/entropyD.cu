/*
 * entropyD.cu
 *
 *  Created on: 11/01/2012
 *      Author: carlos
 ********
 *  Copyright (c) 2012 Carlos Riveros
 *
 *  This file is part of gEpiCount.
 *
 *  gEpiCount is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gEpiCount is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with gEpiCount.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "../proc_gpu.h"
#include "entropy_inline.h"

//#define	DEBUG_KERNEL1_PRINTS

// Entropy computation kernel
__global__ void k1_entropy(IntResultPointers *ptrs, int numRes)
{
	int	resIdx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);
	count_type4	res;
	res.y = ptrs->v[CLASS0][resIdx];
	res.z = ptrs->v[CLASS1][resIdx];
	for (int j = 0; j < NUM_FUNCS; ++j)
	{
		res.w = ptrs->c[CLASS0][resIdx+j*numRes];
		res.x = ptrs->c[CLASS1][resIdx+j*numRes];

		ptrs->ent[resIdx+j*numRes] = entropyDelta4(res);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_KERNEL1_PRINTS)
		if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y < 2)
		{
			printf("\nAt resIdx:%d, FUNC:%d\tc0:%d/%d\tc1:%d/%d, Ent:%f\n", resIdx, j, res.w, res.y, res.x, res.z, ptrs->ent[resIdx+j*numRes]);
		}
#endif
	}
}

#undef DEBUG_KERNEL1_PRINTS




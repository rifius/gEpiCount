/*
 * results_k2.cpp
 *
 * Contains functions auxiliary to management of results for kernel 2
 *
 *  Created on: 15/02/2013
 *      Author: carlos
 *
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
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../inc/gABKEpi.h"
#include "../proc_gpu.h"

/*
 * Allocates and initialises the device buffers to hold results for
 * each invocation of the grid.
 * Returns number of bytes allocated in device or host (the same), or -1 if error.
 */
int allocResults_device(PairInteractionResult **d_pir, int nPIR, const struct _paramP1 &par)
{
	int byteCounter = 0;
	CUDACHECK(cudaMalloc((void**)d_pir,nPIR*sizeof(PairInteractionResult)),par.gpuNum);
	byteCounter += nPIR * sizeof(PairInteractionResult) + 1;


	return byteCounter;
}

int allocResults_host(PairInteractionResult **d_pir, int nPIR, const struct _paramP1 &par)
{
	int byteCounter = 0;
	CUDACHECK(cudaMallocHost((void**)d_pir,nPIR*sizeof(PairInteractionResult)),par.gpuNum);
	byteCounter += nPIR * sizeof(PairInteractionResult) + 1;


	return byteCounter;
}

void freeResults_device(const PairInteractionResult &pir, const struct _paramP1 &par)
{
	CUDACHECK(cudaFree((void*)&pir),par.gpuNum);
}

void freeResults_host(const PairInteractionResult &pir, const struct _paramP1 &par)
{
	CUDACHECK(cudaFreeHost((void*)&pir),par.gpuNum);
}

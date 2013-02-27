/*
 * setup1.h
 *
 *  Created on: 23/02/2012
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

#ifndef SETUP1_H_
#define SETUP1_H_

#include 	<cuda_runtime.h>
#include 	"../../inc/gEpiCount.h"

// Alloc, reorder and send data to device
// This function changes data memory layout to sample mayor order, to
// improve coalescing on the gpu
template<typename T>
__host__ int allocSendData_device(PlinkReader<T> *pr, T **bits, T **valid,
		const struct _paramP1 &par) {
	int nBytes = 0;
	CUDACHECK(cudaMalloc(bits,pr->DataSize()), par.gpuNum);
	CUDACHECK(cudaMalloc(valid,pr->DataSize()), par.gpuNum);
	nBytes = 2 * pr->DataSize();
	T *aux = (T*) malloc(pr->DataSize());
	memset(aux, 0, pr->DataSize());
	for (int j = 0; j < pr->numSNPs(); ++j) {
		for (int k = 0; k < pr->elementsSNP(); ++k) {
			aux[j + k * pr->numSNPs()] =
					pr->BitData()[j * pr->elementsSNP() + k];
		}
	}
	CUDACHECK(cudaMemcpy(*bits,aux,pr->DataSize(),cudaMemcpyHostToDevice),
			par.gpuNum);
	CUDACHECK(cudaDeviceSynchronize(), par.gpuNum);
	memset(aux, 0, pr->DataSize());
	for (int j = 0; j < pr->numSNPs(); ++j) {
		for (int k = 0; k < pr->elementsSNP(); ++k) {
			aux[j + k * pr->numSNPs()] = pr->ValidData()[j * pr->elementsSNP()
					+ k];
		}
	}
	CUDACHECK(cudaMemcpy(*valid,aux,pr->DataSize(),cudaMemcpyHostToDevice),
			par.gpuNum);
	CUDACHECK(cudaDeviceSynchronize(), par.gpuNum);
	free(aux);
	return nBytes;
}

template<typename T>
__host__ void freeData_device(T *bits, T *valid, const struct _paramP1 &par) {
	CUDACHECK(cudaFree(valid), par.gpuNum);
	CUDACHECK(cudaFree(bits), par.gpuNum);
}

template<typename T>
__host__ int allocSendData_device(PlinkReader<T> &pr,
		struct _dataPointersD<T> &dptrs, const struct _paramP1 &par) {
	int nBytes = 0;
	CUDACHECK(cudaMalloc(&dptrs.bits,pr.DataSize()), par.gpuNum);
	CUDACHECK(cudaMalloc(&dptrs.valid,pr.DataSize()), par.gpuNum);
	nBytes = 2 * pr.DataSize();
	// Allocate buffer and place in element major order instead of SNP major order
	T *aux = (T*) malloc(pr.DataSize());
	memset(aux, 0, pr.DataSize());
	for (int j = 0; j < pr.numSNPs(); ++j) {
		for (int k = 0; k < pr.elementsSNP(); ++k) {
			aux[j + k * pr.numSNPs()] = pr.BitData()[j * pr.elementsSNP() + k];
		}
	}
	CUDACHECK(cudaMemcpy(dptrs.bits,aux,pr.DataSize(),cudaMemcpyHostToDevice),
			par.gpuNum);
	CUDACHECK(cudaDeviceSynchronize(), par.gpuNum);
	memset(aux, 0, pr.DataSize());
	for (int j = 0; j < pr.numSNPs(); ++j) {
		for (int k = 0; k < pr.elementsSNP(); ++k) {
			aux[j + k * pr.numSNPs()] =
					pr.ValidData()[j * pr.elementsSNP() + k];
		}
	}
	CUDACHECK(cudaMemcpy(dptrs.valid,aux,pr.DataSize(),cudaMemcpyHostToDevice),
			par.gpuNum);
	CUDACHECK(cudaDeviceSynchronize(), par.gpuNum);
	free(aux);
	return nBytes;
}

template<typename T>
__host__ void freeData_device(struct _dataPointersD<T> &dptrs,
		const struct _paramP1 &par) {
	CUDACHECK(cudaFree(dptrs.valid), par.gpuNum);
	CUDACHECK(cudaFree(dptrs.bits), par.gpuNum);
}

#endif /* SETUP1_H_ */

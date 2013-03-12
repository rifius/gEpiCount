/*
 * setup2.h
 * Builds upon setup1.h
 *
 *  Created on: 17/01/2013
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

#ifndef SETUP2_H_
#define SETUP2_H_

#include	"../proc_gpu.h"
#include	"setup1.h"

#define		MAX_ABKRES_NBR	32767	// Maximum possible number of elements in list
#define		MIN_LISTN_PP	2		// Minimum number of indexes to keep per pair

// Allocs and send to device all input data pertaining to calculation of ABK pairs.
template <typename T, typename Key>
__host__ int allocSendData_device(PlinkReader<T> &pr, ABKEpiGraph<Key, T> &abkEG, ABKInputData<T> &dInDPT, const struct _paramP1 &par)
{
	dInDPT.nELE = pr.elementsSNP();
	dInDPT.numSNPs = pr.numSNPs();
	dInDPT.numSamples = pr.numSamples();
	dInDPT.numPairs = abkEG.nPairs();
	int nBytes = allocSendData_device(pr, dInDPT.dpt, par);

	// TODO: Probably needs to change layout for coalescing
	CUDACHECK(cudaMalloc(&dInDPT.pFlag,dInDPT.numPairs*sizeof(unsigned char)),par.gpuNum);
	nBytes += dInDPT.numPairs;
	CUDACHECK(cudaMalloc(&dInDPT.worstCovers,dInDPT.numPairs*sizeof(int)),par.gpuNum);
	nBytes += dInDPT.numPairs*sizeof(int);

	CUDACHECK(cudaMemcpy(dInDPT.pFlag,abkEG.getPairFlags(),dInDPT.numPairs*sizeof(unsigned char),cudaMemcpyHostToDevice),par.gpuNum);

	return nBytes;
}

template <typename T>
__host__ void freeData_device(ABKInputData<T> &dInDPT, const struct _paramP1 &par)
{
	freeData_device(dInDPT.dpt, par);
	CUDACHECK(cudaFree(dInDPT.pFlag),par.gpuNum);
}

template <typename T, typename Key>
__host__ bool computeResultBufferSize(ABKResultDetails &abkrd, ABKEpiGraph<Key, T> &abkEG, const struct _paramP1 &par)
{
	// Output buffer
	struct cudaDeviceProp dProp;
	std::memset(&dProp,0,sizeof(dProp));
	CUDACHECK(cudaGetDeviceProperties(&dProp,par.gpuNum),par.gpuNum);
	clog << "K2: Device: " << par.gpuNum << " [" << dProp.name << "] " << dProp.major << "." << dProp.minor << endl;
	size_t availMem, totalMem;
	CUDACHECK(cudaMemGetInfo(&availMem,&totalMem),par.gpuNum);

	// How many times does a single interaction result index for all pairs fit ?
	int nDevPLength = availMem / (abkEG.nPairs() * sizeof(short int));
	if (nDevPLength < 1)
	{
		clog << "K2: Insufficient memory on device even for a single edge per pair" << std::endl;
		return false;
	}
	int listn = (availMem - nDevPLength * abkEG.nPairs() * sizeof(short int)) / sizeof(PairInteractionResult);
	while (listn < MAX_ABKRES_NBR && nDevPLength > MIN_LISTN_PP)
	{
		nDevPLength--;
		listn = (availMem - nDevPLength * abkEG.nPairs() * sizeof(short int)) / sizeof(PairInteractionResult);
	}
	if (nDevPLength > abkEG.nResPP())
	{
		nDevPLength = abkEG.nResPP();
	}
	clog << "K2: Memory on device: Global:" << dProp.totalGlobalMem << ", Available: " << (availMem+ONEMEGA-1)/ONEMEGA << "Mb" << endl;
	clog << "K2: ListSize(pair): " << nDevPLength << " ResultListSize: " << listn << endl;
	abkrd.maxSelected = listn;
	abkrd.dResPP = nDevPLength;
	abkrd.currIndex = 0;
	abkrd.nPairs = abkEG.nPairs();
	return true;
}

#endif /* SETUP2_H_ */

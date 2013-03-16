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
//	dInDPT.nEP = dInDPT.nELE * (dInDPT.nELE + 1) / 2;				// Number of pairs of elements
	int nBytes = allocSendData_device(pr, dInDPT.dpt, par);

	// TODO: Probably needs to change layout for coalescing
	CUDACHECK(cudaMalloc(&dInDPT.pFlag,dInDPT.numPairs*sizeof(unsigned char)),par.gpuNum);
	nBytes += dInDPT.numPairs;
	CUDACHECK(cudaMalloc(&dInDPT.worstCovers,dInDPT.numPairs*sizeof(int)),par.gpuNum);
	nBytes += dInDPT.numPairs*sizeof(int);
	CUDACHECK(cudaMalloc(&dInDPT.locks,dInDPT.numPairs*sizeof(unsigned int)),par.gpuNum);
	nBytes += dInDPT.numPairs*sizeof(unsigned int);

	CUDACHECK(cudaMemcpy(dInDPT.pFlag,abkEG.getPairFlags(),dInDPT.numPairs*sizeof(unsigned char),cudaMemcpyHostToDevice),par.gpuNum);

	return nBytes;
}

template <typename T>
__host__ void freeData_device(ABKInputData<T> &dInDPT, const struct _paramP1 &par)
{
	freeData_device(dInDPT.dpt, par);
	CUDACHECK(cudaFree(dInDPT.pFlag),par.gpuNum);
	CUDACHECK(cudaFree(dInDPT.worstCovers),par.gpuNum);
	CUDACHECK(cudaFree(dInDPT.locks),par.gpuNum);
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
	int nDevPLength = availMem / (abkEG.nPairs() * sizeof(PILindex_t));
	if (nDevPLength < 1)
	{
		clog << "K2: Insufficient memory on device even for a single edge per pair" << std::endl;
		return false;
	}
	int listn = (availMem - nDevPLength * abkEG.nPairs() * sizeof(PILindex_t)) / sizeof(PairInteractionResult);
	while (listn < MAX_ABKRES_NBR && nDevPLength > MIN_LISTN_PP)
	{
		nDevPLength--;
		listn = (availMem - nDevPLength * abkEG.nPairs() * sizeof(PILindex_t)) / sizeof(PairInteractionResult);
	}
	if (nDevPLength > abkEG.nResPP())
		nDevPLength = abkEG.nResPP();
	if (listn > MAX_ABKRES_NBR)
		listn = MAX_ABKRES_NBR;

	clog << "K2: Memory on device: Global:" << (dProp.totalGlobalMem+ONEMEGA-1)/ONEMEGA << "Mb, Available: " << (availMem+ONEMEGA-1)/ONEMEGA << "Mb" << endl;
	clog << "K2: ListSize(pair): " << nDevPLength << " Needed: " << (nDevPLength*abkEG.nPairs()*sizeof(PILindex_t)+ONEMEGA-1)/ONEMEGA << "Mb" << endl;
	clog << "K2: ResultListSize: " << listn << " Needed: " << (listn*sizeof(PairInteractionResult)+ONEMEGA-1)/ONEMEGA << "Mb" << endl;
	abkrd.maxSelected = listn;
	abkrd.dResPP = nDevPLength;
	abkrd.currIndex = 0;
	abkrd.nPairs = abkEG.nPairs();
	return true;
}

// Can not reside on another .cu file
template <typename T, typename Key>
__host__ bool sendWorstCovers(ABKEpiGraph<Key,T> &eg, ABKInputData<T> &id, const struct _paramP1 &par)
{
	CUDACHECK(cudaMemcpy(id.worstCovers,eg.getWorstCovers(),eg.nPairs()*sizeof(int),cudaMemcpyHostToDevice),par.gpuNum);
	id.worstCoverAlpha = eg.getWorstAlphaCover();
	id.worstCoverBeta = eg.getWorstBetaCover();
	return false;
}

// Function to initialise the current launch of grid.
// It manages initialisation of the pairList on device, clears the (already copied back) array of results
// on device, initialises members of the structure for results.
template <typename T, typename Key>
__host__ bool initCycleResults_device(const ABKEpiGraph<Key,T> &eg, const ABKInputData<T> &id, const ABKResultDetails &ld_abr,
		ABKResultDetails *p_abkr, const struct _paramP1 &par)
{
#define	HERE_BLOCK_DIM	8
#define	HERE_GRID_DIM	256
	Timer tt;
//	// Dispatch init kernel
//	dim3 block(HERE_BLOCK_DIM,1,1), grid(HERE_GRID_DIM,1,1);
//	k2_initPairList<<<grid,block,16384>>>(p_abkr);
	// Clear results array on device
	CUDACHECK(cudaMemset(ld_abr.selected,0,ld_abr.maxSelected*sizeof(PairInteractionResult)),par.gpuNum);
	CUDACHECK(cudaMemset(ld_abr.pairList,EMPTY_INDEX_B1,ld_abr.nPairs*ld_abr.dResPP*sizeof(PILindex_t)),par.gpuNum);
	CUDACHECK(cudaMemset(id.locks,EMPTY_INDEX_B1,id.numPairs*sizeof(unsigned int)),par.gpuNum);
	CUDACHECK(cudaDeviceSynchronize(),par.gpuNum);
	clog << "K2: initCycle " << tt.stop() << "sec. " << p_abkr << endl;
#undef 	HERE_BLOCK_DIM
#undef	HERE_GRID_DIM
	return true;
}


#endif /* SETUP2_H_ */

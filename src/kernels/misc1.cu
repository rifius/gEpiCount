/*
 * misc1.cpp
 *
 * Miscelaneous auxiliary host functions for kernel1
 *
 *  Created on: 23/01/2012
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
#include <cuda_runtime.h>

#include "../../inc/gEpiCount.h"
#include "../proc_gpu.h"

// If commented, CUDA mallocs memory for host buffers (pinned)
//#define 	MALLOC_OUTPUT_BUFFER

// Nasty function trying to get a reasonable grid size
__host__ bool computeBlockGridSize(int nSNPs, int &nIRmax, dim3 &block, dim3 &grid, dim3 &altBlk,
		dim3 &altGrd, const struct _paramP1 &par) {
	// Output buffer
	struct cudaDeviceProp dProp;
	memset(&dProp, 0, sizeof(dProp));
	CUDACHECK(cudaGetDeviceProperties(&dProp,par.gpuNum), par.gpuNum);
	clog << "K1: Device: " << par.gpuNum << " [" << dProp.name << "] " << dProp.major << "."
			<< dProp.minor << endl;
	size_t availMem, totalMem;
	CUDACHECK(cudaMemGetInfo(&availMem,&totalMem), par.gpuNum);
	nIRmax = availMem / (SIZEOF_GPURESULT + SIZEOF_EXTRASORTGPU);
	int nThreadSide = (int) sqrt((float) nIRmax);
	if (nIRmax > par.maxGridBlocks * ONEMEGA) {
		nIRmax = par.maxGridBlocks * ONEMEGA;
		nThreadSide = (int) sqrt((float) nIRmax);
		clog << "K1: Result buffer maximum set to " << nIRmax << " elements, " << nThreadSide
				<< endl;
	}
	else {
		clog << "K1: Result buffer memory limit to " << nIRmax << " elements, " << nThreadSide
				<< endl;
	}
	int nGrdX = 2 * (nThreadSide / par.blkSzX / 2);		// We want them even
	int nGrdY = 2 * (nThreadSide / par.blkSzY / 2);
	nIRmax = nGrdX * nGrdY * par.blkSzX * par.blkSzY;
	clog << "K1: Max Global: " << dProp.totalGlobalMem << ", Available: "
			<< (availMem + ONEMEGA - 1) / ONEMEGA << "MB. num results (nIRmax): " << nIRmax << endl;

	dim3 nBside;
	nBside.x = (nSNPs + par.blkSzX - 1) / par.blkSzX;
	nBside.y = (nSNPs + par.blkSzY - 1) / par.blkSzY;
	int nTotBlocks = nBside.x * (nBside.y + 1) / 2;		// WARN: not sure is this amount exactly
	block = dim3(par.blkSzX, par.blkSzY);
	grid = dim3(nGrdX, nGrdY);
	fprintf(stderr,
			"K1: Block(%d,%d) threads/Block:%d, Blocks/data side:%d,%d, total Blocks:%d, th/grid side:%d,%d grid size:%d,%d, blocks/grid: %d\n",
			par.blkSzX, par.blkSzY, (par.blkSzX * par.blkSzY), nBside.x, nBside.y, nTotBlocks,
			grid.x * block.x, grid.y * block.y, grid.x, grid.y, (grid.x * grid.y));
	if (nBside.x * nBside.y < nGrdX * nGrdY) {
		nGrdX = nBside.x;
		nGrdY = nBside.y;
		grid.x = nGrdX;
		grid.y = nGrdY;
		nIRmax = nGrdX * nGrdY * par.blkSzX * par.blkSzY;
		clog << "K1: Grid size reduced to " << nGrdX << "x" << nGrdY << ", " << nIRmax << " results"
				<< endl;
	}
	altBlk = block;
	altGrd = grid;
	if (par.alt && (block.x * block.y) <= 256 && altGrd.x % 2 == 0) {
		altBlk.x = 32;
		altGrd.x = altGrd.x / 2;
	}
	return true;
}

// Local function to do memory allocation on device
static __host__ bool _mallocResults_device(IntResultPointers &d_irptrs, int nRes, int nSamp,
		dim3 blk, const struct _paramP1 &par) {
	size_t nBy = 0;
	CUDACHECK(cudaMalloc(&(d_irptrs.c[CLASS0]),NUM_FUNCS*nRes*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMalloc(&(d_irptrs.c[CLASS1]),NUM_FUNCS*nRes*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMalloc(&(d_irptrs.v[CLASS0]),nRes*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMalloc(&(d_irptrs.v[CLASS1]),nRes*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMalloc(&d_irptrs.ent,NUM_FUNCS*nRes*sizeof(float)), par.gpuNum);
	CUDACHECK(cudaMalloc(&d_irptrs.idx,NUM_FUNCS*nRes*sizeof(int)), par.gpuNum);
	nBy += nRes	* (NUM_CLASSES * (NUM_FUNCS + 1) * sizeof(count_type)
					+ NUM_FUNCS * (sizeof(float) + sizeof(int)));
	return nBy > 0;
}

// Allocate on device and copy to constant memory struct.
__host__ bool allocResults_device(IntResultPointers &d_irptrs, int nRes, int nSamp, dim3 blk,
		const struct _paramP1 &par) {
	bool ok = _mallocResults_device(d_irptrs, nRes, nSamp, blk, par);
	// FIXME: resP is not being used on device, struct passed as kernel arg
	CUDACHECK(cudaMemcpyToSymbol(resP,&d_irptrs,sizeof(IntResultPointers),0,cudaMemcpyHostToDevice),
			par.gpuNum);
	CUDACHECK(cudaGetLastError(), par.gpuNum);
	return ok;
}

// Allocate on device, prepare and copy to device structure of pointers
__host__ bool allocResults_device(IntResultPointers &d_irptrs, IntResultPointers **pd_irptr,
		int nRes, int nSamp, dim3 blk, const struct _paramP1 &par) {
	CUDACHECK(cudaMalloc(pd_irptr,sizeof(IntResultPointers)), par.gpuNum);
	bool ok = _mallocResults_device(d_irptrs, nRes, nSamp, blk, par);
	CUDACHECK(cudaMemcpy(*pd_irptr,&d_irptrs,sizeof(IntResultPointers),cudaMemcpyHostToDevice),
			par.gpuNum);
	CUDACHECK(cudaGetLastError(), par.gpuNum);
	return ok;
}

// Allocate memory on host for results
// nRes is the number of results being computed on each kernel invocation
// nTop is the number of results of the running list
__host__ bool allocResults_host(IntResultPointers &h_irptrs, int nRes, const struct _paramP1 &par) {
	size_t nBy = 0;
#ifndef		MALLOC_OUTPUT_BUFFER
	CUDACHECK(cudaMallocHost(&(h_irptrs.c[CLASS0]),NUM_FUNCS*nRes*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMallocHost(&(h_irptrs.c[CLASS1]),NUM_FUNCS*nRes*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMallocHost(&(h_irptrs.v[CLASS0]),nRes*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMallocHost(&(h_irptrs.v[CLASS1]),nRes*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMallocHost(&h_irptrs.ent,NUM_FUNCS*nRes*sizeof(float)), par.gpuNum);
	CUDACHECK(cudaMallocHost(&h_irptrs.idx,NUM_FUNCS*nRes*sizeof(int)), par.gpuNum);
#else
	h_irptrs.c[CLASS0] = (count_type *) malloc(NUM_FUNCS*nRes*sizeof(count_type));
	h_irptrs.c[CLASS1] = (count_type *) malloc(NUM_FUNCS*nRes*sizeof(count_type));
	h_irptrs.v[CLASS0] = (count_type *) malloc(nRes*sizeof(count_type));
	h_irptrs.v[CLASS1] = (count_type *) malloc(nRes*sizeof(count_type));
	h_irptrs.ent = (float *) malloc(NUM_FUNCS*nRes*sizeof(float));
	h_irptrs.idx = (int *) malloc(NUM_FUNCS*nRes*sizeof(int));
#endif
	// Aux buffer on host
	h_irptrs.aux = (InteractionResult*) malloc(par.topmost * sizeof(InteractionResult));
	nBy += nRes
			* (NUM_CLASSES * (NUM_FUNCS + 1) * sizeof(count_type)
					+ NUM_FUNCS * (sizeof(float) * sizeof(int)))
			+ par.topmost * sizeof(InteractionResult);
	// Zero
	zeroResults(h_irptrs, nRes, par.topmost);

	return nBy > 0;
}

// Local function to free device resources
static __host__ void _freeResults_device(const IntResultPointers &d_irptrs,
		const struct _paramP1 &par) {
	CUDACHECK(cudaFree(d_irptrs.c[CLASS0]), par.gpuNum);
	CUDACHECK(cudaFree(d_irptrs.c[CLASS1]), par.gpuNum);
	CUDACHECK(cudaFree(d_irptrs.v[CLASS0]), par.gpuNum);
	CUDACHECK(cudaFree(d_irptrs.v[CLASS1]), par.gpuNum);
	CUDACHECK(cudaFree(d_irptrs.ent), par.gpuNum);
	CUDACHECK(cudaFree(d_irptrs.idx), par.gpuNum);
}

// Free device resources
__host__ void freeResults_device(const IntResultPointers &d_irptrs, const struct _paramP1 &par) {
	_freeResults_device(d_irptrs, par);
}

// Free device resources
__host__ void freeResults_device(const IntResultPointers &d_irptrs, IntResultPointers *p_irptrs,
		const struct _paramP1 &par) {
	CUDACHECK(cudaFree(p_irptrs), par.gpuNum);
	_freeResults_device(d_irptrs, par);
}

// Free host resources
__host__ void freeResults_host(const IntResultPointers &h_irptrs, const struct _paramP1 &par) {
#ifndef		MALLOC_OUTPUT_BUFFER
	CUDACHECK(cudaFreeHost(h_irptrs.c[CLASS0]), par.gpuNum);
	CUDACHECK(cudaFreeHost(h_irptrs.c[CLASS1]), par.gpuNum);
	CUDACHECK(cudaFreeHost(h_irptrs.v[CLASS0]), par.gpuNum);
	CUDACHECK(cudaFreeHost(h_irptrs.v[CLASS1]), par.gpuNum);
	CUDACHECK(cudaFreeHost(h_irptrs.ent), par.gpuNum);
	CUDACHECK(cudaFreeHost(h_irptrs.idx), par.gpuNum);
#else
	free(h_irptrs.c[CLASS0]);
	free(h_irptrs.c[CLASS1]);
	free(h_irptrs.v[CLASS0]);
	free(h_irptrs.v[CLASS1]);
	free(h_irptrs.ent);
	free(h_irptrs.idx);
#endif
	free(h_irptrs.aux);
}

// Get results from device.  If the requested number of results is less than the results on GPU
// only numtopmost values and indexes are copied back.
__host__ bool transferBack(const IntResultPointers &d_irptrs, IntResultPointers &h_irptrs,
		int numRes, const struct _paramP1 &par, int nobj) {
	int nsorted = nobj < NUM_FUNCS * numRes ? nobj : NUM_FUNCS * numRes;
	CUDACHECK(cudaMemcpy(h_irptrs.c[CLASS0],d_irptrs.c[CLASS0],NUM_FUNCS*numRes*sizeof(count_type),cudaMemcpyDeviceToHost),
			par.gpuNum);
	CUDACHECK(cudaMemcpy(h_irptrs.c[CLASS1],d_irptrs.c[CLASS1],NUM_FUNCS*numRes*sizeof(count_type),cudaMemcpyDeviceToHost),
			par.gpuNum);
	CUDACHECK(cudaMemcpy(h_irptrs.v[CLASS0],d_irptrs.v[CLASS0],numRes*sizeof(count_type),cudaMemcpyDeviceToHost),
			par.gpuNum);
	CUDACHECK(cudaMemcpy(h_irptrs.v[CLASS1],d_irptrs.v[CLASS1],numRes*sizeof(count_type),cudaMemcpyDeviceToHost),
			par.gpuNum);
	CUDACHECK(cudaMemcpy(h_irptrs.ent,d_irptrs.ent,nsorted*sizeof(float),cudaMemcpyDeviceToHost),
			par.gpuNum);
	CUDACHECK(cudaMemcpy(h_irptrs.idx,d_irptrs.idx,nsorted*sizeof(int),cudaMemcpyDeviceToHost),
			par.gpuNum);
	return true;
}
__host__ bool transferBack(const IntResultPointers &d_irptrs, IntResultPointers &h_irptrs,
		int numRes, const struct _paramP1 &par) {
	return transferBack(d_irptrs, h_irptrs, numRes, par, par.topmost);
}

// Merge results from the running list and current batch of results from device.
// To merge results, we scan linearly the stored results in tops and the stored results in tops
// creating a consolidated list in h_irptrs.aux.  When finished, copy back to tops.
// h_irptrs contains arrays of size numRes plus aux buffer of size numtop
// tops 	contains an array of size numtop
// offTL	has the offset coordinates for current batch of data
// blk,grd	are the block and grid sizes
__host__ bool mergeResults(IntResultPointers &h_irptrs, dim3 offTL, dim3 blk, dim3 grd,
		InteractionResult *tops, int numtop, int numRes) {
	int resCnt = 0;
	int devCnt = 0;
	int topCnt = 0;
	while (resCnt < numtop) {
		if (devCnt == NUM_FUNCS * numRes || tops[topCnt].ent <= h_irptrs.ent[devCnt]) {
			h_irptrs.aux[resCnt] = tops[topCnt];
			resCnt++;
			topCnt++;
		}
		else if (devCnt < NUM_FUNCS * numRes) {
			int ik = h_irptrs.idx[devCnt];
			int iv = ik % numRes;
			short fun = ik / numRes;
			h_irptrs.aux[resCnt].ent = h_irptrs.ent[devCnt];
			h_irptrs.aux[resCnt].n.w = h_irptrs.c[CLASS0][ik];
			h_irptrs.aux[resCnt].n.x = h_irptrs.c[CLASS1][ik];
			h_irptrs.aux[resCnt].n.y = h_irptrs.v[CLASS0][iv];
			h_irptrs.aux[resCnt].n.z = h_irptrs.v[CLASS1][iv];
			h_irptrs.aux[resCnt].fun = fun;
			h_irptrs.aux[resCnt].sA = SNPA_FROM_RINDEX_(iv, blk, grd, offTL);
			h_irptrs.aux[resCnt].sB = SNPB_FROM_RINDEX_(iv, blk, grd, offTL);
			resCnt++;
			devCnt++;
		}
	}
	clog << "Merge: " << (resCnt - topCnt) << endl;
	memcpy(tops, h_irptrs.aux, numtop * sizeof(InteractionResult));
	return true;
}

// Zero host results buffers
__host__ void zeroResults(IntResultPointers &h_irptrs, int nRes, int nTop) {
	memset(h_irptrs.c[CLASS0], 0, NUM_FUNCS * nRes * sizeof(count_type));
	memset(h_irptrs.c[CLASS1], 0, NUM_FUNCS * nRes * sizeof(count_type));
	memset(h_irptrs.v[CLASS0], 0, nRes * sizeof(count_type));
	memset(h_irptrs.v[CLASS1], 0, nRes * sizeof(count_type));
	memset(h_irptrs.ent, 0, NUM_FUNCS * nRes * sizeof(float));
	memset(h_irptrs.idx, 0, NUM_FUNCS * nRes * sizeof(int));
	memset(h_irptrs.aux, 0, nTop * sizeof(InteractionResult));
}

// Initialization kernel.  Prepares the list of results for next invocation.
// To be used in conjunction with cudaMemset()
__global__ void k1_init1(IntResultPointers ptrs, int numRes) {
	int resIdx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y
			+ (threadIdx.y * blockDim.x + threadIdx.x);
	for (int j = 0; j < NUM_FUNCS; ++j) {
		ptrs.idx[resIdx + j * numRes] = resIdx + j * numRes;
	}
}

__host__ void initDeviceResults(IntResultPointers &d_irptrs, dim3 grid, dim3 block, int nir,
		const struct _paramP1 &par) {
	CUDACHECK(cudaMemset(d_irptrs.c[CLASS0],0,NUM_FUNCS*nir*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMemset(d_irptrs.c[CLASS1],0,NUM_FUNCS*nir*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMemset(d_irptrs.v[CLASS0],0,nir*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMemset(d_irptrs.v[CLASS1],0,nir*sizeof(count_type)), par.gpuNum);
	CUDACHECK(cudaMemset(d_irptrs.ent,0,NUM_FUNCS*nir*sizeof(float)), par.gpuNum);
	k1_init1<<<grid,block>>>(d_irptrs, nir);
}

// Auxiliary function to emulate all indexes and check derived indexes.  Pretty lame.
// To be used in lieu of the kernel
// WARN: It writes the h_irptrs arrays to detect memory overwrites
__host__ void checkLaunchLoops(dim3 gr, dim3 bl, size_t numRes, dim3 firstG,
		IntResultPointers &h_irptrs, size_t nSNPs, size_t nELE) {
	char flag[5];
	flag[3] = 0;
	for (int bx = 0; bx < gr.x; bx++) {
		for (int by = 0; by < gr.y; by++) {
			for (int tx = 0; tx < bl.x; tx++) {
				for (int ty = 0; ty < bl.y; ty++) {
					//===== This scope is same as kernel
					// bx, by are block coordinates
					// tx, ty are thread coordinates
					if (bx > by) continue;
					flag[0] = flag[1] = flag[2] = ' ';
					int idxA = by * bl.y + ty + firstG.y;// SNP A goes on the Y direction ("vertical", "row")
					int idxB = bx * bl.x + tx + firstG.x;
					int resIdx = (by * gr.x + bx) * bl.x * bl.y + (ty * bl.x + tx);
					if (resIdx < numRes) {
						count_type4 z, zz;
						zz.w = bx;
						zz.x = by;
						zz.y = tx;
						zz.z = ty;
						if (h_irptrs.idx[resIdx] != 0) {
							// We use the values for BFUNC_AND to store block and thread
							z.w = h_irptrs.c[CLASS0][NUM_FUNCS * resIdx];
							z.x = h_irptrs.c[CLASS1][NUM_FUNCS * resIdx];
							z.y = h_irptrs.v[CLASS0][resIdx];
							z.z = h_irptrs.v[CLASS1][resIdx];
							int sA = SNPA_FROM_RINDEX_(h_irptrs.idx[resIdx], bl, gr, firstG);
							int sB = SNPB_FROM_RINDEX_(h_irptrs.idx[resIdx], bl, gr, firstG);
							clog << "Idx " << resIdx << " written:" << endl;
							clog << "  Prev: bx,by,tx,ty: " << z.w << "," << z.x << "," << z.y
									<< "," << z.z << " sA,sB:" << sA << "," << sB << " idx:"
									<< h_irptrs.idx[resIdx] << endl;
							clog << "  Curr: bx,by,tx,ty: " << zz.w << "," << zz.x << "," << zz.y
									<< "," << zz.z << " sA,sB:" << idxA << "," << idxB << " idx:"
									<< resIdx << endl;
						}
						h_irptrs.c[CLASS0][NUM_FUNCS * resIdx] = zz.w;
						h_irptrs.c[CLASS1][NUM_FUNCS * resIdx] = zz.x;
						h_irptrs.v[CLASS0][resIdx] = zz.y;
						h_irptrs.v[CLASS1][resIdx] = zz.z;
						h_irptrs.idx[resIdx] = resIdx;
					}
					if (idxA > nSNPs || idxB > idxA || resIdx >= numRes) continue;
					if (idxA >= nSNPs) {
						flag[0] = '@';
						clog << flag << "Bx,By: " << bx << "," << by << "\tTx,Ty: " << tx << ","
								<< ty << "\tsA,sB: " << idxA << "," << idxB << "\tresIdx: "
								<< resIdx << endl;
						continue;
					}
					if (idxB >= idxA) {
						flag[1] = '>';
						clog << flag << "Bx,By: " << bx << "," << by << "\tTx,Ty: " << tx << ","
								<< ty << "\tsA,sB: " << idxA << "," << idxB << "\tresIdx: "
								<< resIdx << endl;
						continue;
					}
					if (resIdx >= numRes) {
						flag[2] = '!';
						clog << flag << "Bx,By: " << bx << "," << by << "\tTx,Ty: " << tx << ","
								<< ty << "\tsA,sB: " << idxA << "," << idxB << "\tresIdx: "
								<< resIdx << endl;
						continue;
					}
				}
			}
		}
	}
}

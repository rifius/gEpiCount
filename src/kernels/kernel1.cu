/*
 * kernel1.cu
 *
 *  Created on: 05/01/2012
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

#include <cuda.h>

//#define		DEBUG_KERNEL1_PRINTS

#include "../../inc/gEpiCount.h"
#include "../misc/Timer.h"
#include "../misc/print_misc.h"
#include "../proc_gpu.h"
#include "setup1.h"

//#define		SORT_ON_HOST

#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include	<iterator>

#ifdef	__CUDACC__
/*
 * Memory on device.  A pointer to all the reserved shared memory space
 * the structure with information about pointers and two symbols for masks
 * in constant memory
 */
__device__ __constant__ 	ui8	d_mask0[CONST_MASK_MAX_SIZE];
__device__ __constant__ 	ui8	d_mask1[CONST_MASK_MAX_SIZE];
__device__ __constant__	IntResultPointers	resP;
extern __shared__ count_type _sm_buffer[];
#endif

static const string ffunNames[FFUNC_LAST + 1] = { "ENTROPY", "FISHER PV", "Unknown" };

// Kernel function for process 1, counting bits part
// firstG contains the column and row coordinates of the first thread of current grid in x, y.
// The structure containing pointers to the arrays of results is in constant memory
template<typename T>
__global__ void k1_count_sm(const struct _dataPointersD<T> dptrs, IntResultPointers *ptrs,
		int numRes, dim3 firstG, int nSNPs, int nELE) {
	int idxA = blockIdx.y * blockDim.y + firstG.y + threadIdx.y; // SNP A goes on the Y direction ("vertical", "row")
	int idxB = blockIdx.x * blockDim.x + firstG.x + threadIdx.x;

	count_type vv[NUM_CLASSES];
	count_type cc[NUM_CLASSES][NUM_FUNCS];
	vv[CLASS0] = 0;
	vv[CLASS1] = 0;
	for (int k = 0; k < NUM_FUNCS; ++k) {
		cc[CLASS0][k] = 0;
		cc[CLASS1][k] = 0;
	}

	bool doit = (idxA < nSNPs && idxB <= idxA);

	T *bA = dptrs.bits + idxA;
	T *bB = dptrs.bits + idxB;
	T *vA = dptrs.valid + idxA;
	T *vB = dptrs.valid + idxB;
	T v, c, m0, m1;

	// For non-square blocks, we take the smallest dimension as the number of elements for each row & col
	// to copy to shared mem.  The shared mem matrices are of dimension:
	//   blockDim.y * ns for bA, vA
	//   blockDim.x * ns for bB, vB
	short ns = blockDim.x < blockDim.y ? blockDim.x : blockDim.y;
	T *bA_s, *bB_s, *vA_s, *vB_s;
	bA_s = (T*) (_sm_buffer);
	bB_s = bA_s + (ns * blockDim.y);
	vA_s = bB_s + (ns * blockDim.x);
	vB_s = vA_s + (ns * blockDim.y);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_KERNEL1_PRINTS)
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
		printf("ptrs %p, idxA:%d, idxB:%d, ns:%d, doit:%d\n", ptrs, idxA, idxB, ns, doit);
	}
#endif

	// Loop over number of times a whole SNP is blocked in shared mem
	for (int m = 0; m < ((nELE + ns - 1) / ns); m++) {
		__syncthreads();
		if (threadIdx.x < ns) {
			// Elements of each row consecutive in memory
			bA_s[threadIdx.y * ns + threadIdx.x] = bA[(m * ns + threadIdx.x) * nSNPs];
			vA_s[threadIdx.y * ns + threadIdx.x] = vA[(m * ns + threadIdx.x) * nSNPs];
			// Elements of consecutive rows consecutive in memory
//			bA_s[threadIdx.y+blockDim.y*threadIdx.x] = bA[(m * ns + threadIdx.x) * nSNPs];
//			vA_s[threadIdx.y+blockDim.y*threadIdx.x] = vA[(m * ns + threadIdx.x) * nSNPs];
		}
		if (threadIdx.y < ns) {
			// Elements of each column consecutive in memory
			bB_s[threadIdx.x * ns + threadIdx.y] = bB[(m * ns + threadIdx.y) * nSNPs];
			vB_s[threadIdx.x * ns + threadIdx.y] = vB[(m * ns + threadIdx.y) * nSNPs];
			// Elements of consecutive column consecutive in memory
//			bB_s[threadIdx.x+blockDim.x*threadIdx.y] = bB[(m * ns + threadIdx.y) * nSNPs];
//			vB_s[threadIdx.x+blockDim.x*threadIdx.y] = vB[(m * ns + threadIdx.y) * nSNPs];
		}
		__syncthreads();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_KERNEL1_PRINTS)
		if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
		{
			printf("Debug SM (%d,%d)(%d,%d) nE:%d m:%d ns:%d ok:%d A:%d B:%d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,nELE,m,ns,doit,idxA,idxB);
			printf("\nAt 0,0 debug print\n");
			for (int j = 0; j < ns; ++j)
			{
				printf("m:%2d j:%2d  %016lx  %016lx  %016lx  %016lx\n", m, j, bA_s[j], bB_s[j], vA_s[j], vB_s[j]);
			}
		}
#endif

		for (int j = 0; j < (nELE - m * ns > ns ? ns : nELE - m * ns); ++j) {
			int sj = m * ns + j;
			v = vA_s[threadIdx.y * ns + j] & vB_s[threadIdx.x * ns + j];			// Valid bits

			m0 = maskElem<T>(d_mask0, sj);
			m1 = maskElem<T>(d_mask1, sj);
			vv[CLASS0] += __popcll(v & m0);
			vv[CLASS1] += __popcll(v & m1);

			// BFUNC_AND
			c = (bA_s[threadIdx.y * ns + j] & bB_s[threadIdx.x * ns + j]);
			cc[CLASS0][BFUNC_AND] += __popcll(c & v & m0);
			cc[CLASS1][BFUNC_AND] += __popcll(c & v & m1);
			// BFUNC_NAND
			c = ((~bA_s[threadIdx.y * ns + j]) & bB_s[threadIdx.x * ns + j]);
			cc[CLASS0][BFUNC_NAND] += __popcll(c & v & m0);
			cc[CLASS1][BFUNC_NAND] += __popcll(c & v & m1);
			// BFUNC_ANDN
			c = (bA_s[threadIdx.y * ns + j] & (~bB_s[threadIdx.x * ns + j]));
			cc[CLASS0][BFUNC_ANDN] += __popcll(c & v & m0);
			cc[CLASS1][BFUNC_ANDN] += __popcll(c & v & m1);
			// BFUNC_NANDN (implemented as ~(a | b)
			c = (~(bA_s[threadIdx.y * ns + j] | bB_s[threadIdx.x * ns + j]));
			cc[CLASS0][BFUNC_NANDN] += __popcll(c & v & m0);
			cc[CLASS1][BFUNC_NANDN] += __popcll(c & v & m1);
			// BFUNC_XOR
			c = (bA_s[threadIdx.y * ns + j] ^ bB_s[threadIdx.x * ns + j]);
			cc[CLASS0][BFUNC_XOR] += __popcll(c & v & m0);
			cc[CLASS1][BFUNC_XOR] += __popcll(c & v & m1);
		}
	}
	int resIdx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y
			+ (threadIdx.y * blockDim.x + threadIdx.x);
	if (doit) {
		ptrs->v[CLASS0][resIdx] = vv[CLASS0];
		ptrs->v[CLASS1][resIdx] = vv[CLASS1];
		for (int k = 0; k < NUM_FUNCS; ++k) {
			ptrs->c[CLASS0][resIdx + k * numRes] = cc[CLASS0][k];
			ptrs->c[CLASS1][resIdx + k * numRes] = cc[CLASS1][k];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_KERNEL1_PRINTS)
			if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y < 2 && firstG.y == 0 && firstG.x == 0)
			{
				printf("\nAt idxA:%d, idxB:%d, FUNC:%d\tc0:%d/%d\tc1:%d/%d\n", idxA, idxB, k, cc[CLASS0][k], vv[CLASS0], cc[CLASS1][k], vv[CLASS1]);
			}
#endif
		}
	}
}

template<typename T>
__global__ void k1_count(const struct _dataPointersD<T> dptrs, IntResultPointers *ptrs, int numRes,
		dim3 firstG, int nSNPs, int nELE) {
	int idxA = blockIdx.y * blockDim.y + threadIdx.y + firstG.y;// SNP A goes on the Y direction ("vertical", "row")
	int idxB = blockIdx.x * blockDim.x + threadIdx.x + firstG.x;
	int resIdx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y
			+ (threadIdx.y * blockDim.x + threadIdx.x);
	count_type vv[NUM_CLASSES];
	count_type cc[NUM_CLASSES][NUM_FUNCS];

	if (idxA >= nSNPs || idxB > idxA || resIdx >= numRes) return;
	T *bA = dptrs.bits + idxA;
	T *bB = dptrs.bits + idxB;
	T *vA = dptrs.valid + idxA;
	T *vB = dptrs.valid + idxB;

	vv[CLASS0] = 0;
	vv[CLASS1] = 0;
	for (int k = 0; k < NUM_FUNCS; ++k) {
		cc[CLASS0][k] = 0;
		cc[CLASS1][k] = 0;
	}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_KERNEL1_PRINTS)
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && firstG.y == 0 && firstG.x == 0)
	{
		printf("ptrs %p\n", ptrs);
		printf("\nAt 0,0 debug print: idxA:%d, idxB:%d\n", idxA, idxB);
		printf("\n---- Data bits bA\n");
		for (int k = 0; k < nELE; ++k)
		{
			printf("%016lx ", bA[k*nSNPs]);
			if (k % 4 == 3)	printf("\n");
		}
		printf("\n");
		for (int k = 0; k < nELE; ++k)
		{
			printf("%016lx ", bA[1+k*nSNPs]);
			if (k % 4 == 3)	printf("\n");
		}
		printf("\n");
		printf("\n---- Data valid vA\n");
		for (int k = 0; k < nELE; ++k)
		{
			printf("%016lx ", vA[k*nSNPs]);
			if (k % 4 == 3)	printf("\n");
		}
		printf("\n");
		for (int k = 0; k < nELE; ++k)
		{
			printf("%016lx ", vA[1+k*nSNPs]);
			if (k % 4 == 3)	printf("\n");
		}
		printf("\n");
	}
#endif

	T c, v, m0, m1;
	for (int j = 0; j < nELE; ++j) {
		int idxj = j * nSNPs;
		v = vA[idxj] & vB[idxj];								// Valid bits
		m0 = maskElem<T>(d_mask0, j);
		m1 = maskElem<T>(d_mask1, j);
		vv[CLASS0] += __popcll(v & m0);
		vv[CLASS1] += __popcll(v & m1);

		// BFUNC_AND
		c = (bA[idxj] & bB[idxj]);
		cc[CLASS0][BFUNC_AND] += __popcll(c & v & m0);
		cc[CLASS1][BFUNC_AND] += __popcll(c & v & m1);
		// BFUNC_NAND
		c = ((~bA[idxj]) & bB[idxj]);
		cc[CLASS0][BFUNC_NAND] += __popcll(c & v & m0);
		cc[CLASS1][BFUNC_NAND] += __popcll(c & v & m1);
		// BFUNC_ANDN
		c = (bA[idxj] & (~bB[idxj]));
		cc[CLASS0][BFUNC_ANDN] += __popcll(c & v & m0);
		cc[CLASS1][BFUNC_ANDN] += __popcll(c & v & m1);
		// BFUNC_NANDN (implemented as ~(a | b)
		c = (~(bA[idxj] | bB[idxj]));
		cc[CLASS0][BFUNC_NANDN] += __popcll(c & v & m0);
		cc[CLASS1][BFUNC_NANDN] += __popcll(c & v & m1);
		// BFUNC_XOR
		c = (bA[idxj] ^ bB[idxj]);
		cc[CLASS0][BFUNC_XOR] += __popcll(c & v & m0);
		cc[CLASS1][BFUNC_XOR] += __popcll(c & v & m1);
	}
	{
		ptrs->v[CLASS0][resIdx] = vv[CLASS0];
		ptrs->v[CLASS1][resIdx] = vv[CLASS1];
		for (int k = 0; k < NUM_FUNCS; ++k) {
			ptrs->c[CLASS0][resIdx + k * numRes] = cc[CLASS0][k];
			ptrs->c[CLASS1][resIdx + k * numRes] = cc[CLASS1][k];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_KERNEL1_PRINTS)
			if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y < 2 && firstG.y == 0 && firstG.x == 0)
			{
				printf("\nAt idxA:%d, idxB:%d, resIdx:%d, FUNC:%d\tc0:%d/%d\tc1:%d/%d\n", idxA, idxB, resIdx, k,
						cc[CLASS0][k], vv[CLASS0], cc[CLASS1][k], vv[CLASS1]);
			}
#endif
		}
	}
}

static __host__
inline int computeShmSize(dim3 blk, int eleSz) {
	int sz = 0;
	int ns = blk.x < blk.y ? blk.x : blk.y;
	sz += 2 * ns * eleSz * (blk.x + blk.y);		// bX & vX
	return sz;
}

// Launcher of the kernel 1 function.
template<typename T>
__host__ static void launch_process_gpu_1(PlinkReader<T> *pr, InteractionResult *tops,
		const struct _paramP1 &par) {
	struct _dataPointersD<T> d_DPtrs;
	size_t byteCounter;
	double totTime = 0.0f;
	int nIRmax;
	dim3 block, grid, altBlk, altGrd;

	Timer tt;
	CUDACHECK(cudaSetDevice(par.gpuNum), par.gpuNum);
	byteCounter = allocSendData_device(*pr, d_DPtrs, par);
	clog << "K1: Input+Aux Data: " << byteCounter << std::endl;

	// Get size of grid
	computeBlockGridSize(pr->numSNPs(), nIRmax, block, grid, altBlk, altGrd, par);

	// To improve coalescing and do on-device sorting, we split the InteractionResult data elements into separate arrays
	IntResultPointers d_IRPtrs, h_IRPtrs, *p_IRPtrs;
	allocResults_device(d_IRPtrs, &p_IRPtrs, nIRmax, pr->numSamples(), block, par);
	allocResults_host(h_IRPtrs, nIRmax, par);

	thrust::device_ptr<float> dev_entf(d_IRPtrs.ent);
	thrust::device_ptr<int> dev_idxs(d_IRPtrs.idx);

	size_t availMem, totalMem;
	CUDACHECK(cudaMemGetInfo(&availMem,&totalMem), par.gpuNum);
	totTime += tt.stop();
	clog << "K1: Alloc time " << totTime << ", Available mem < "
			<< (availMem + ONEMEGA - 1) / ONEMEGA << "MB" << endl;

	tt.start();

	// Copy class masks to constant memory
	if (pr->elementsSNP()*sizeof(T) > CONST_MASK_MAX_SIZE*sizeof(ui8))
	{
		cerr << "K1: Class Masks do not fit in constant memory.  Change compilation size" << endl;
		return;
	}
	// Debug
//	print_data(pr,0,2);
	CUDACHECK(cudaMemcpyToSymbol(d_mask0,pr->classData(1)->mask(),pr->elementsSNP()*sizeof(T),0,cudaMemcpyHostToDevice),par.gpuNum);
	CUDACHECK(cudaMemcpyToSymbol(d_mask1,pr->classData(2)->mask(),pr->elementsSNP()*sizeof(T),0,cudaMemcpyHostToDevice),par.gpuNum);

	int shmSize = computeShmSize(block, pr->ELEMENT_SIZE);
	if (par.shmem) {
		if (shmSize > 3 * 16384) {
			cerr << "K1: Shared Memory kernel exceeds available SHM size, " << shmSize
					<< ". Reduce block size." << endl;
			return;
		}
		else if (shmSize > 16000)
		CUDACHECK(cudaFuncSetCacheConfig(k1_count_sm<T>,cudaFuncCachePreferShared), par.gpuNum);
		clog << "K1: Starting MULTI KERNEL w/ SHARED MEM mode, element major order.  SHM size "
				<< shmSize << endl;
	}
	else {
		clog << "K1: Starting MULTI KERNEL mode, element major order" << endl;
		shmSize = 0;
	}

	clog << "K1: Objective function is " << ffunNames[par.ffun] << endl;

	{
		cudaEvent_t ev_start;
		cudaEvent_t ev_end1;
		cudaEvent_t ev_end2;
		cudaEvent_t ev_end3;
		cudaEvent_t ev_end4;
		cudaEvent_t ev_end5;
		cudaEvent_t ev_end6;
		bool primed = false;
		// We want to compute nSNPs * nSNPs interactions in blocks of dimension BLOCK_DIM x BLOCK_DIM, considering
		// only the lower triangular part of the interaction matrix is computed

		CUDACHECK(cudaEventCreate(&ev_start), par.gpuNum);
		CUDACHECK(cudaEventCreate(&ev_end1), par.gpuNum);
		CUDACHECK(cudaEventCreate(&ev_end2), par.gpuNum);
		CUDACHECK(cudaEventCreate(&ev_end3), par.gpuNum);
		CUDACHECK(cudaEventCreate(&ev_end4), par.gpuNum);
		CUDACHECK(cudaEventCreate(&ev_end5), par.gpuNum);
		CUDACHECK(cudaEventCreate(&ev_end6), par.gpuNum);
		float tim_sort = 0.0f;
		float tim_copy = 0.0f;
		double tim_cycle = 0.0;
		double tim_merge = 0.0;
		double tim_total = 0.0;
		float tim_kern = 0.0f;
		float tim_kern_a = 0.0f;
		float tim_kern_b = 0.0f;

		dim3 fgr(0, 0, 0), old_fgr(0, 0, 0);
		const unsigned int nny = grid.y * block.y;
		const unsigned int nnx = grid.x * block.x;
		for (fgr.y = 0; fgr.y < pr->numSNPs(); fgr.y += nny) {
			for (fgr.x = 0; fgr.x <= fgr.y; fgr.x += nnx) {
				clog << "K1: init <<(" << altGrd.x << "," << altGrd.y << "," << altGrd.z << "),("
						<< altBlk.x << "," << altBlk.y << "," << altBlk.z << ")>>\t";
				clog << "count <<("	<< grid.x << "," << grid.y << "," << grid.z << "),("
						<< block.x << "," << block.y << "," << block.z << ")," << shmSize << ">>\t";
				clog << "[objective] <<(" << grid.x << "," << grid.y << "," << grid.z << "),("
						<< block.x << "," << block.y << "," << block.z << ")>>\t";
				clog << "fgr=" << fgr.x << "," << fgr.y << "\tObjective from: "
						<< h_IRPtrs.aux[0].ent << " to: " << h_IRPtrs.aux[par.topmost - 1].ent
						<< endl;

				Timer tx1, tx2;
				CUDACHECK(cudaEventRecord(ev_start,0), par.gpuNum);

				initDeviceResults(d_IRPtrs, altGrd, altBlk, nIRmax, par);
				CUDACHECK(cudaEventRecord(ev_end2,0), par.gpuNum);
				if (par.shmem)
					k1_count_sm<T> <<<grid,block,shmSize>>>(d_DPtrs, p_IRPtrs, nIRmax, fgr, pr->numSNPs(), pr->elementsSNP());
				else
					k1_count<T><<<grid,block>>>(d_DPtrs, p_IRPtrs, nIRmax, fgr, pr->numSNPs(), pr->elementsSNP());
				CUDACHECK(cudaEventRecord(ev_end3,0), par.gpuNum);
				if (par.ffun == FFUNC_ENTROPY)
					k1_entropy<<<grid,block>>>(p_IRPtrs, nIRmax);
				else if (par.ffun == FFUNC_FISHERPV)
					k1_fisherpv<<<grid,block>>>(p_IRPtrs, nIRmax, pr->numSamples());
				CUDACHECK(cudaEventRecord(ev_end1,0), par.gpuNum);

				if (primed) {
					Timer tx3;
					mergeResults(h_IRPtrs, old_fgr, block, grid, tops, par.topmost, nIRmax);
					tim_merge = tx3.stop() * 1000.0;
					// DEBUG:  Status printing at diagonal points in the matrix
					if (fgr.x == 0) {
						float nt = (float) pr->numSNPs() / (float) (fgr.y - 1);	// Fraction so far is 1/nt
						nt = nt * nt;
						float rem = (nt - 1.0f) * tim_total;
						float pct = 100.0f / nt;
						string uu = "sec";
						if (rem > 3600.0) {
							rem /= 3600.0;
							uu = "hr";
						}
						clog << "K1: Total time " << tim_total << "sec (" << pct << "%)"
								<< " Remaining: " << rem << uu << endl;
						if (par.intermediate) {
							clog << "K1: Last processed: " << (fgr.y - 1) << "," << (fgr.y - 1)
									<< endl;
							printInteractions(pr, tops, par.topmost, PRINT_STRIDE, PRINT_CONTEXT);
						}
					}
				}

#ifndef	SORT_ON_HOST
				CUDACHECK(cudaEventRecord(ev_end4,0), par.gpuNum);
				// Sort in place of objectivey values and indexes.
				// TODO: thrust::sort uses radix sort from bc40. It allocates and deallocates memory on each invocation (for keys and values)
				// Extra time and performance gain can be obtained switching to bc40 code and repeatedly using aux mem.  It's just difficult.
				// WARN: The other arrays must be accessed by index !
				thrust::sort_by_key(dev_entf, dev_entf + (NUM_FUNCS * nIRmax), dev_idxs);
				CUDACHECK(cudaEventRecord(ev_end5,0), par.gpuNum);
				CUDACHECK(cudaEventSynchronize(ev_end5), par.gpuNum);
#endif	/* ifndef SORT_ON_HOST	*/

				transferBack(d_IRPtrs, h_IRPtrs, nIRmax, par);
				CUDACHECK(cudaEventRecord(ev_end6,0), par.gpuNum);
				CUDACHECK(cudaEventSynchronize(ev_end6), par.gpuNum);
				// These events should have been already passed after the DeviceSynchroinize
				CUDACHECK(cudaEventElapsedTime(&tim_kern,ev_start,ev_end1), par.gpuNum);
				CUDACHECK(cudaEventElapsedTime(&tim_kern_a,ev_start,ev_end2), par.gpuNum);
				CUDACHECK(cudaEventElapsedTime(&tim_kern_b,ev_end2,ev_end3), par.gpuNum);
				CUDACHECK(cudaEventElapsedTime(&tim_sort,ev_end4,ev_end5), par.gpuNum);
				CUDACHECK(cudaEventElapsedTime(&tim_copy,ev_end5,ev_end6), par.gpuNum);
				old_fgr = fgr;

#ifdef	SORT_ON_HOST
				tx1.start();
				std::vector<float>::iterator e_begin(h_IRPtrs.ent);
				std::vector<int>::iterator i_begin(h_IRPtrs.idx);
				thrust::sort_by_key(e_begin, e_begin+(NUM_FUNCS*nIRmax), i_begin);
				tim_sort = tx1.stop() * 1000.0;
#endif
				tim_cycle = tx2.stop() * 1000.0;
				primed = true;
//				printResults(h_IRPtrs, nIRmax, block, grid, old_fgr, PRINT_STRIDE);	// DEBUG:

				tim_total = tt.stop();
				clog << "K1: Time: total: " << tim_total << "  cycle: " << tim_cycle << "  kern: "
						<< tim_kern << ": i: " << tim_kern_a << ", c:" << tim_kern_b << ", e:"
						<< (tim_kern - tim_kern_a - tim_kern_b) << "  sort: " << tim_sort
						<< "  copy: " << tim_copy << "  merge: " << tim_merge << "  overlap:"
						<< ((int) ((tim_kern + tim_sort + tim_copy) / tim_cycle * 1000.0)) / 10.0
						<< endl;
			}
		}

		mergeResults(h_IRPtrs, old_fgr, block, grid, tops, par.topmost, nIRmax);
//		printResults(h_IRPtrs, nIRmax, block, grid, old_fgr, PRINT_STRIDE);	// DEBUG:

		CUDACHECK(cudaEventDestroy(ev_start), par.gpuNum);
		CUDACHECK(cudaEventDestroy(ev_end1), par.gpuNum);
		CUDACHECK(cudaEventDestroy(ev_end2), par.gpuNum);
		CUDACHECK(cudaEventDestroy(ev_end3), par.gpuNum);
		CUDACHECK(cudaEventDestroy(ev_end4), par.gpuNum);
		CUDACHECK(cudaEventDestroy(ev_end5), par.gpuNum);
		CUDACHECK(cudaEventDestroy(ev_end6), par.gpuNum);
	}

	double mt = tt.stop();
	clog << "K1: Copy+Run time " << mt << " Total: " << mt + totTime << endl;

//	printResults(tops, numTopmost, 1000);

	freeResults_host(h_IRPtrs, par);
	freeResults_device(d_IRPtrs, par);
	freeData_device(d_DPtrs, par);
}

// Functions called in host app
__host__ void process_gpu_1(PlinkReader<ui8> *pr, InteractionResult *tops,
		const struct _paramP1 &par) {
	launch_process_gpu_1<ui8>(pr, tops, par);
}

__host__ void process_gpu_1(PlinkReader<ui4> *pr, InteractionResult *tops,
		const struct _paramP1 &par) {
//	launch_process_gpu_1<ui4>(pr, tops, par);
}

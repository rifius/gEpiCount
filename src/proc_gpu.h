/*
 * proc_gpu.h
 *
 *  Created on: 29/12/2011
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

#ifndef PROC_GPU_H_
#define PROC_GPU_H_

#include "../inc/gEpiCount.h"
#include "../inc/gABKEpi.h"
#include "misc/Timer.h"
#include "ABKEpiGraph.h"

#define		K1_FREE_FACTOR		0.8f		// Fraction of available memory to be used in results (max)
// Size of results computed per GPU thread
#define		SIZEOF_GPURESULT	((NUM_FUNCS+1)*NUM_CLASSES*sizeof(count_type)+NUM_FUNCS*(sizeof(float)+sizeof(int)))
#define		SIZEOF_EXTRASORTGPU	(NUM_FUNCS*(sizeof(float)+sizeof(int)+1))		// +1 is heuristic for some additional space

/*
 * Memory on device.  A pointer to all the reserved shared memory space
 * the structure with information about pointers and two symbols for masks
 * in constant memory
 */
extern __device__ __constant__ 	ui8	d_mask0[CONST_MASK_MAX_SIZE];
extern __device__ __constant__ 	ui8	d_mask1[CONST_MASK_MAX_SIZE];
extern __device__ __constant__	IntResultPointers	resP;
extern __shared__ count_type _sm_buffer[];

// Input and precomputed data for ABK.  Same as for counts
// plus the flags indicating if pair is valid, alpha or beta.
template <typename T>
class ABKInputData
{
public:
//	const int 	SAMPLES_ELEMENT;
	int		numPairs;
	int		numSNPs;
	int 	numSamples;
	int 	nELE;
//	int		nEP;				// Number of element pairs. Auxiliary
//	int		nIter;				// Number of iterations per thread.  Auxiliary
	int		worstCoverAlpha;	// worst from all alpha covers
	int		worstCoverBeta;		// worst from all beta covers
	struct _dataPointersD<T>	dpt;
	unsigned char 				*pFlag;
	int		*worstCovers;		// worst for each pair
	unsigned int	*locks;		// To lock result list elements

//	ABKInputData(): SAMPLES_ELEMENT(8 * sizeof(T)) {};
};

/*
 * Indices management.
 * Results are accommodated in layers of size
 *   nres = (blk.x * blk.y * grd.x * grd.y),
 * where blk and grd are the respective block and grid sizes.
 * A given thread and offset codes the indices for SNPs A and B as:
 *   idxA = bid.y * blk.y + tid.y + off.y
 *   idxB = bid.x * blk.x + tid.x + off.x
 * and the corresponding index is in multiples of
 *   ridx = bid.y * grd.x * blk.x * blk.y +		// Full block rows in grid
 *        + bid.x         * blk.x * blk.y +		// Current (incomplete) row
 *        + tid.y * blk.x +						// Full block rows
 *        + tid.x
 * Result for a function j is stored at position:
 *   j * nres + ridx.
 * The inverse formulas to "decode" from an index the different parameters are given below:
 * Notice that the 'geometry' of the problem must be given: block and grid sizes, offset of grid.
 * Function:
 *   fun = index / (blk.x * blk.y * grd.x * grd.y)
 * Index of SNP A/B:
 *   idx = index % (blk.x * blk.y * grd.x * grd.y)
 *   idxA = (((idx/(grd.x*blk.x*blk.y))*blk.y) + ((idx%(blk.x*blk.y))/blk.x) + off.y)
 *   idxB = ((((idx%(grd.x*blk.x*blk.y))/(blk.x*blk.y))*blk.x) + (idx%blk.x) + off.x)
 * Block and Thread Id:
 *   bid.x = ((index % (blk.x * blk.y * grd.x * grd.y)) % (grd.x * blk.x * blk.y)) / (blk.x * blk.y)
 *   bid.y = (index % (blk.x * blk.y * grd.x * grd.y)) / (grd.x * blk.x * blk.y)
 *   tid.x = (index % (blk.x * blk.y * grd.x * grd.y)) % blk.x
 *   tid.y = ((index % (blk.x * blk.y * grd.x * grd.y)) % (blk.x * blk.y)) / blk.x
 *
 */
__host__ __device__
inline int SNPA_FROM_RINDEX_(int idx, dim3 blk, dim3 grd, dim3 off)		// This is the Y coordinate
{
	idx = idx % (blk.x * blk.y * grd.x * grd.y);
	return (((idx/(grd.x*blk.x*blk.y))*blk.y) + ((idx%(blk.x*blk.y))/blk.x) + off.y);
}

__host__ __device__
inline int SNPB_FROM_RINDEX_(int idx, dim3 blk, dim3 grd, dim3 off)		// This is the X coordinate
{
	idx = idx % (blk.x * blk.y * grd.x * grd.y);
	return ((((idx%(grd.x*blk.x*blk.y))/(blk.x*blk.y))*blk.x) + (idx%blk.x) + off.x);
}

__host__ __device__
inline int FUNC_FROM_RINDEX_(int idx, dim3 blk, dim3 grd)
{
	return (idx / (blk.x * blk.y * grd.x * grd.y));
}

__host__ __device__
inline int BIDX_FROM_RINDEX_(int idx, dim3 blk, dim3 grd)
{
	return ((idx % (blk.x * blk.y * grd.x * grd.y)) % (grd.x * blk.x * blk.y)) / (blk.x * blk.y);
}

__host__ __device__
inline int BIDY_FROM_RINDEX_(int idx, dim3 blk, dim3 grd)
{
	return (idx % (blk.x * blk.y * grd.x * grd.y)) / (grd.x * blk.x * blk.y);
}

__host__ __device__
inline int TIDX_FROM_RINDEX_(int idx, dim3 blk, dim3 grd)
{
	return (idx % (blk.x * blk.y * grd.x * grd.y)) % blk.x;
}

__host__ __device__
inline int TIDY_FROM_RINDEX_(int idx, dim3 blk, dim3 grd)
{
	return ((idx % (blk.x * blk.y * grd.x * grd.y)) % (blk.x * blk.y)) / blk.x;
}

template <typename T>
__forceinline__ __device__ T maskElem(T *ptr, int j)
{
	return (T) *(((T*)ptr)+j);
}

// Circular Rotate right by one
template <typename T>
__host__ __device__ inline T rotr1(const T val)
{
	return (val >> 1) | (val << (sizeof(T)*8 - 1));
}

// Flags for pairs
#define		P_VALID		0x1
#define		P_ALPHA		0x2

// Indices for counters
#define		IDX_ALPHA	0
#define		IDX_BETA	1

//######## For kernel 1
void process_gpu_1(PlinkReader<ui8> *pr, InteractionResult *tops, const struct _paramP1 &par);
void process_gpu_1(PlinkReader<ui4> *pr, InteractionResult *tops, const struct _paramP1 &par);

bool computeBlockGridSize(int nSNPs, int &nIRmax, dim3 &block, dim3 &grid, dim3 &altBlk, dim3 &altGrd, const struct _paramP1 &par);

bool allocResults_device(IntResultPointers &d_irptrs, int nRes, int nSamp, dim3 blk, const struct _paramP1 &par);
bool allocResults_device(IntResultPointers &d_irptrs, IntResultPointers **pd_irptr, int nRes, int nSamp, dim3 blk, const struct _paramP1 &par);
bool allocResults_host(IntResultPointers &h_irptrs, int nRes, const struct _paramP1 &par);
void freeResults_device(const IntResultPointers &d_irptrs, const struct _paramP1 &par);
void freeResults_device(const IntResultPointers &d_irptrs, IntResultPointers *p_irptrs, const struct _paramP1 &par);
void freeResults_host(const IntResultPointers &h_irptrs, const struct _paramP1 &par);

bool transferBack(const IntResultPointers &d_irptrs, IntResultPointers &h_irptrs, int numRes, const struct _paramP1 &par, int nobj);
bool transferBack(const IntResultPointers &d_irptrs, IntResultPointers &h_irptrs, int numRes, const struct _paramP1 &par);
bool mergeResults(IntResultPointers &h_IRPtrs, dim3 offTL, dim3 blk, dim3 grd, InteractionResult *tops, int numTopmost, int nIRmax);
void zeroResults(IntResultPointers &h_irptrs, int nRes, int nTop);
void initDeviceResults(IntResultPointers &d_irptrs, dim3 grid, dim3 block, int nir, const struct _paramP1 &par);

void computeEntropyObjective(IntResultPointers dptrs, dim3 grd, dim3 blk, int nRes, int nSamp);

void checkLaunchLoops(dim3 gr, dim3 bl, size_t numRes, dim3 firstG, IntResultPointers &h_irptrs, size_t nSNPs, size_t nELE);

__global__ void k1_fisherpv(IntResultPointers *ptrs, int numRes, int nSamp);
__global__ void k1_entropy(IntResultPointers *ptrs, int numRes);
template <typename T>
__global__ void k1_count_sm(const struct _dataPointersD<T> dptrs, IntResultPointers *ptrs, int numRes, dim3 firstG, int nSNPs, int nELE);
template <typename T>
__global__ void k1_count(const struct _dataPointersD<T> dptrs, IntResultPointers *ptrs, int numRes, dim3 firstG, int nSNPs, int nELE);

//######## For Kernel 2 (ABK)
void process_gpu_2(PlinkReader<ui8> *pr, ABKEpiGraph<int64_t, ui8> &abkeg, const struct _paramP1 &par);

template <typename T>
__global__ void k2_g(const ABKInputData<T> P, const dim3 firstG, const ABKResultDetails *ptr);
template <typename T, class Funct>
__device__ void k2_macro(const ABKInputData<T> P, int idxA, int idxB, int boolfunc, const ABKResultDetails *pr, Funct op);
template <typename T>
__global__ void k2_v2(const ABKInputData<T> P, const dim3 firstG, const ABKResultDetails *pr);
__global__ void k2_initPairList(ABKResultDetails *pr);

int allocResults_device(const ABKResultDetails &ld_abkr, ABKResultDetails **pd_abkr, const struct _paramP1 &par);
int allocResults_host(ABKResultDetails &h_abkr, const struct _paramP1 &par);
void freeResults_device(const ABKResultDetails &h_abkr, const ABKResultDetails *p_abkr, const struct _paramP1 &par);
void freeResults_host(const ABKResultDetails &h_pir, const struct _paramP1 &par);

bool transferBack(const ABKResultDetails *pd_abk, ABKResultDetails *h_abk, const struct _paramP1 &par);

//template <typename Key, typename T>
//bool sendWorstCovers(const ABKEpiGraph<Key,T> &eg, const ABKInputData<T> &id, const struct _paramP1 &par);
// Misc functions defined as templates but specific instantiation is indicated here
//bool sendWorstCovers(const ABKEpiGraph<int64_t,ui8> &eg, const ABKInputData<ui8> &id, const struct _paramP1 &par);
//bool initCycleResults_device(const ABKEpiGraph<int64_t,ui8> &eg, const ABKInputData<ui8> &id, const ABKResultDetails &ld_abr,
//		ABKResultDetails *p_abkr, const struct _paramP1 &par);
//template <typename Key, typename T>
//bool initCycleResults_device(const ABKEpiGraph<Key,T> &eg, const ABKInputData<T> &id, const ABKResultDetails &ld_abr,
//		ABKResultDetails *p_abkr, const struct _paramP1 &par);

#endif /* PROC_GPU_H_ */

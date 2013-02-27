/*
 * gEpiCount.h
 *
 *  Created on: 20/12/2011
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

#ifndef GEPICOUNT_H_
#define GEPICOUNT_H_

#include <cstdio>
#include <string>
#include <cuda.h>
#include <builtin_types.h>

#include "../src/reader/PlinkReader.h"
#include "../src/reader/BinPlinkReader.h"

#define CUDACHECK(a,g)	\
	{ cudaError_t b = (a); if(b != cudaSuccess) fprintf(stderr, "E[%d]: Error in %s (%s:%d): (%d) %s\n", (int)(g), #a,  __FILE__, __LINE__, b, cudaGetErrorString(b)); }

#define		ONEMEGA		(1024*1024)

typedef unsigned int   			ui4;
typedef unsigned long long int  ui8;

// These two must be consistently defined with the same base type
//typedef short int		count_type;
//typedef short4			count_type4;
typedef int				count_type;
typedef int4			count_type4;

#define		CONST_MASK_MAX_SIZE		128		// This is 1KB if ui8 elements, 8Kbits

#define		NUM_FUNCS		5
#define		NUM_CLASSES		2
#define		CLASS0			0
#define		CLASS1			1

typedef enum BinaryFunction
{
	BFUNC_AND,
	BFUNC_NAND,
	BFUNC_ANDN,
	BFUNC_NANDN,
	BFUNC_XOR,
	BFUNC_LAST
} BinaryFunction;

typedef enum FitnessFunction
{
	FFUNC_ENTROPY,
	FFUNC_FISHERPV,
	FFUNC_LAST
} FitnessFunction;

typedef struct _resultI
{
	int		sA;		// SNP A
	int		sB;		// SNP B
	count_type4	n;
	float	ent;	// Entropy
	char	fun;
} InteractionResult;

// Input data for counts calculation.  Array Pointers on device
template <typename T>
struct _dataPointersD
{
	T		*bits;	// binary SNP value
	T		*valid;	// valid bits for data
};

// The interactions are computed and stored in GPU in the following way:
// We have NUM_CLASSES classes (for the time being this is hard coded as 2)
// and there is no redundant information.
//
// NUM_CLASSES arrays of length N for counts of valid bits
// NUM_CLASSES arrays of length NUM_FUNCS*N for counts of 'on' bits
// One array of length NUM_FUNCS*N of entropy (or pval)
// One array of length NUM_FUNCS*N of indices in previous.
//
// Function and SNP indices are implicit from the index:
// index % NUM_FUNCS is the function.
// index / NUM_FUNCS is the index in valid counts
typedef struct _resultPointersD
{
	count_type	*v[NUM_CLASSES];
	count_type	*c[NUM_CLASSES];
	float		*ent;
	// Index for indirect sorting on GPU
	int			*idx;
	// Only used in host
	InteractionResult	*aux;
} IntResultPointers;

// Parameters for kernel 1 invocation
struct _paramP1
{
	int		gpuNum;			// GPU #
	int		topmost;		// Number of interactions to print
	bool	gold;			// Do gold OR gpu
	bool	shmem;			// Use SHM memory kernel
	bool	alt; 			// Use alternate block & grid on init & entropy computation
	bool	intermediate;	// Print periodic sample results at diagonal
	string	name;			// Input file root
	int 	blkSzX;			// Obscure
	int		blkSzY;			// idem
	int		maxGridBlocks;	// idem
	FitnessFunction ffun;	// Implemented kernels
};

//========== Prototypes

extern "C" {
float fisher_pval(int n11, int n12, int n21, int n22);
};

#endif /* GEPICOUNT_H_ */

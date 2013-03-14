/*
 * gABKEpi.h
 *
 *  Created on: 01/03/2013
 *      Author: carlos
 *
 *  Definitions for gpu ABK Epistasis instance generator
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
 *
 */

#ifndef GABKEPI_H_
#define GABKEPI_H_

#include <cmath>
#include "gEpiCount.h"

// Individual result for an interaction across sample pairs.
// It provides the cover on the alpha and beta sides, and the
// entropy / p-value [not implemented yet]
// TODO: For now we put explicitly the SNP indexes and func,
// but should be encoded in a single int64 to save some space and better
// struct alignment
typedef struct _resultP
{
	int		sA;		// SNP A
	int		sB;		// SNP B
//	int4	n;
//	float	ent;	// Entropy
	int		alphaC;	// Alpha side cover
	int		betaC;	// Beta side cover
	char	fun;
} PairInteractionResult;

// For the global list of results kept in the host, we do
// reference counting
typedef struct _trackedResultP
{
	PairInteractionResult	ir;
	int		refCount;
} TrackedPairInteractionResult;

// The collection of results for a (series) of grid launches.
// Contains a list of PairInteractionResult, with a counter of current
// used elements, and the list of nPairs x dResPP indexes to the
// previous list
// In this initial version we do not keep track of reference counting
// so once a PairInteractionResult makes it to the list, it will remain
// there even if subsequent additions eliminate all links from pairs to
// this particular result
// This structure can not be passed by value to the kernel.  It
// must be in global memory, and the counter incremented atomically.
typedef	ui2					PILindex_t;		// So this type can be changed selectively
typedef struct _abkResultDetails
{
	int						maxSelected;	// Capacity of the 'selected' list
	int						currIndex;		// Current last element used
	int						nPairs;
	short int 				dResPP;			// How many ResPP we keep here
	PairInteractionResult	*selected;
	PILindex_t 				*pairList;		// Indexes per pair
} ABKResultDetails;


// A couple of macro function definitions to convert from a row, column (col > row)
// to a unique index in a squared matrix of size N, upper triangular part.
// And also back from the index to the row, column.
// row, col in [0,N), TriIndex in [0,N*(N-1)/2)
// The diagonal elements are not included.  If Diagonal is needed (col >= row), then
// the N argument has to be M+1, where M is the size of the matrix.
// WARNING, The back function writes to the arguments.
// Defined as macro so can be used INLINE anywhere

// Version with row and column in [1,N], col >= row, index starting at 1 in the diagonal, index in [1,N*(N+1)/2]
#define	DiagIndexAtRow_V1D(row,N)			((N+1)*((row)-1)-((row)*((row)-1)/2)+1)
#define	ROWCOLATINDEX_V1D(idx,N,row,col)	{ \
		row = floor(((double)(2*N+3)-sqrt((double)(2*N+3)*(double)(2*N+3)-8.0*(double)(N+idx)))/2.0);	\
		col = idx-((N+1)*((row)-1)-((row)*((row)-1)/2)+1)+row;	\
}
#define	INDEXATROWCOL_V1D(row,col,N)		(DiagIndexAtRow_V1D(row,N)-row+col)


// Version with row and column in [0,N), col >= row, index in the diagonal starting at 0, index in [0,N*(N+1)/2)
#define	DiagIndexAtRow_V0D(row,N)			((N+1)*(row)-((row)*((row)+1)/2))
#define	ROWCOLATINDEX_V0D(idx,N,row,col)	{ \
		row = floor(((double)(2*N+3)-sqrt((double)(2*N+3)*(double)(2*N+3)-8.0*(double)(N+idx+1)))/2.0-1);	\
		col = idx-((N+1)*(row)-((row)*((row)+1)/2)+1)+row+1;	\
}
#define	INDEXATROWCOL_V0D(row,col,N)		(DiagIndexAtRow_V0D(row,N)-row+col)


// Version with row and column in [0,N), col > row, index does not include diagonal, index in [0,N*(N-1)/2)
#define	DiagIndexAtRow_V00ND(row,N)			((N)*(row)-((row)*((row)+1)/2))
#define	ROWCOLATINDEX_V00ND(idx,N,row,col)	{ \
		row = floor(((double)(2*N+1)-sqrt((double)(2*N+1)*(double)(2*N+1)-8.0*(double)(N+idx)))/2.0-1);	\
		col = idx-((N)*(row)-((row)*((row)+1)/2))+row+1;	\
}
#define	INDEXATROWCOL_V00ND(row,col,N)		(DiagIndexAtRow_V00ND(row,N)-row+col-1)


#define		TriIndexFromRC(row,col,N)	(((N)-1)*(row)-(row)*((row)+1)/2+(col)-1)
#define		RCFromTriIndex(idx,N,row,col)	{ float a = (float)(2*(N)-1)*(float)(2*(N)-1) - (float)(8*((idx)-(N)+1));	\
											  a = ((float)(2*(N)-1) - sqrtf(a))/2.0;		\
											  row = (int) ceilf(a); col = (idx) - (row) * (N) + (row) * ((row) -1) / 2; }


#endif /* GABKEPI_H_ */

/*
 * gABKEpi.h
 *
 *  Created on: 01/03/2013
 *      Author: carlos
 *
 *  Definitions for gpu ABK Epistasis instance generator
 *
 *
 */

#ifndef GABKEPI_H_
#define GABKEPI_H_

#include "gEpiCount.h"

// Input and precomputed data for ABK.  Same as for counts
// plus the flags indicating if pair is valid, alpha or beta.
template <typename T>
struct _dataPointersPairsD
{
	struct _dataPointersD<T>	dpt;
	unsigned char 				*pFlag;
};

// Individual result for an interaction across sample pairs.
// It provides the cover on the alpha and beta sides, and the
// entropy / p-value [not implemented yet]
typedef struct _resultP
{
	int		sA;		// SNP A
	int		sB;		// SNP B
//	int4	n;
	float	ent;	// Entropy
	int		alphaC;	// Alpha side cover
	int		betaC;	// Beta side cover
	short	fun;
} PairInteractionResult;

// Current results for each grid dispatch, against which the
// interactions are judged:
// the value of worst cover and the array of worst cover per pair
typedef struct _currentPairThresholds
{
	int		worstCoverAlpha;	// worst from all alpha covers
	int		worstCoverBeta;		// worst from all beta covers
	int		*worstCovers;		// worst for each pair
} CurrentPairCovers;

// The results for pairs.
// Contains the mode of results (either dense bitpack or sparse list of pairs
// (in the later case how many valid pairs), and the interaction result
typedef struct _abkResultDetails
{

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

/*
 * ancillary2.h
 *
 * Aux template functions for kernel 2 code
 *
 *  Created on: 15/03/2013
 *      Author: carlos
 */

#ifndef ANCILLARY2_H_
#define ANCILLARY2_H_

#include "../../inc/gEpiCount.h"

template <typename T>
//__forceinline__
__device__ T validBitsElement(const struct _dataPointersD<T> P, int idxA, int idxB, int ele, int nSNPs)
{
	T	res = 0;
	if (idxA >= nSNPs || idxB >= nSNPs)
		return res;
	T		*vA = P.valid + idxA;
	T		*vB = P.valid + idxB;
	int idxj = ele * nSNPs;
	res = vA[idxj] & vB[idxj];								// Valid bits
	res &= (maskElem<T>(d_mask0, ele) | maskElem<T>(d_mask1, ele));
	return res;
}

template <typename T>
//__forceinline__
__device__ T binaryFuncElement(const struct _dataPointersD<T> P, int idxA, int idxB, int fun, int ele, int nSNPs)
{
	T	res = 0;
	if (idxA >= nSNPs || idxB >= nSNPs)
		return res;
	T		*bA = P.bits + idxA;
	T		*bB = P.bits + idxB;

	int idxj = ele * nSNPs;

	switch(fun)
	{
	case BFUNC_AND:
		res = (bA[idxj] & bB[idxj]);
//		cc[CLASS0][BFUNC_AND] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_AND] += __popcll(c & v & m1);
		break;
	case BFUNC_NAND:
		res = ((~bA[idxj]) & bB[idxj]);
//		cc[CLASS0][BFUNC_NAND] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_NAND] += __popcll(c & v & m1);
		break;
	case BFUNC_ANDN:
		res = (bA[idxj] & (~bB[idxj]));
//		cc[CLASS0][BFUNC_ANDN] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_ANDN] += __popcll(c & v & m1);
		break;
	case BFUNC_NANDN:	// (implemented as ~(a | b)
		res = (~(bA[idxj] | bB[idxj]));
//		cc[CLASS0][BFUNC_NANDN] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_NANDN] += __popcll(c & v & m1);
		break;
	case BFUNC_XOR:
		res = (bA[idxj] ^ bB[idxj]);
//		cc[CLASS0][BFUNC_XOR] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_XOR] += __popcll(c & v & m1);
		break;
	default:
		break;
	}
	return res;
}



#endif /* ANCILLARY2_H_ */

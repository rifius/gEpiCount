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

// Function to precompute the flags per pair and the total number of alpha / beta pairs.
// Returns a uint2 tuple with alpha & beta counts.
template <typename T>
__host__ static uint2 precompute_pairs(const PlinkReader<T> &pr, unsigned char *flags)
{
	const size_t SAMPLES_ELEMENT = 8 * sizeof(T);
	uint2	res;
	res.x = res.y = 0;		// x is alpha, y is beta
	size_t  pairIndex = 0;

#if defined(DEBUG_PRINTS)
//	cerr << "  pairIndex\t=\ts1,s2 [e1,e2]\tv:valid f:flags\t[s1.Name,s1.Sex,s1.Class]\t[s2.Name,s2.Sex,s2.Class]" << std::endl;
#endif

	Timer tt;
	int nEP = (pr.elementsSNP() * (pr.elementsSNP() + 1)) / 2;
	for (int epi = 0; epi < nEP; epi++)
	{
		int e1, e2;
#ifdef ALTERNATE_IMPLEMENTATION
		// Find element row
		e1 = 2 * pr.elementsSNP() - 1;
		{
			float a = (float) e1 * (float) e1 - 8.0 * (float) (epi - pr.elementsSNP() + 1);
			a = ((float) e1 - sqrtf(a)) / 2.0;
			e1 = (int) ceilf(a);
		}
		// and element column
		e2 = epi - e1 * (pr.elementsSNP() - 1) + (e1 * (e1 - 1)) / 2;
#else
		ROWCOLATINDEX_V0D(epi,pr.elementsSNP(),e1,e2);
#endif

		// Valid and class masks
		T ms0p1 = (pr.classData(1)->mask())[e1];	// Class 0 element 1
		T ms1p1 = (pr.classData(2)->mask())[e1];	// Class 1 element 1
		T ms0p2 = (pr.classData(1)->mask())[e2];	// Class 0 element 2
		T ms1p2 = (pr.classData(2)->mask())[e2];	// Class 1 element 2

//		for (int bitgap = 0; bitgap < SAMPLES_ELEMENT; bitgap++)
//			...
//			for (int n = 0; n < SAMPLES_ELEMENT; n++)
//			...
//				sIdy = e1 * SAMPLES_ELEMENT + n;
//				sIdx = e2 * SAMPLES_ELEMENT + (n + bitgap) % SAMPLES_ELEMENT;
//
//				if (sIdx < nSamp && sIdy < nSamp && sIdx > sIdy)
//				{
//					...
//					pairIndex++;


		for (size_t bitgap = 0; bitgap < SAMPLES_ELEMENT; bitgap++)
		{
			T equal = (ms0p1 & ms0p2) | (ms1p1 & ms1p2);	// Bits on indicate pair is of equal class (beta)
			T diffe = (ms0p1 & ms1p2) | (ms1p1 & ms0p2);	// Bits on indicate pair is of different classes (alpha)
			T selector = (T) 0x1;
			if (e1 != e2 || bitgap > 0)
			{
				for (size_t n = 0; n < SAMPLES_ELEMENT; n++)
				{
					size_t s1 = e1 * SAMPLES_ELEMENT + n;
					size_t s2 = e2 * SAMPLES_ELEMENT + (n + bitgap) % SAMPLES_ELEMENT;
					if (s1 < pr.numSamples() && s2 < pr.numSamples() && s2 > s1)
					{
						pairIndex = INDEXATROWCOL_V00ND(s1,s2,pr.numSamples());
						bool eqeq = ((equal & selector) == selector);
						bool dfdf = ((diffe & selector) == selector);
						bool vv = eqeq || dfdf;
						if (vv)
						{
							flags[pairIndex] = P_VALID;
							if (eqeq)
								res.y++;
							else	// is alpha
							{
								flags[pairIndex] |= P_ALPHA;
								res.x++;
							}
						}
#if defined(DEBUG_PRINTS)
//						Sample spl1 = pr.getSample(s1);
//						Sample spl2 = pr.getSample(s2);
//						cerr << "  " << pairIndex << "\t=\t" << s1 << "," << s2 << " [" << e1 << "," << e2 << "]" <<
//								"\tv:" << vv << " f:" << (int) flags[pairIndex] <<
//								"\t[" << spl1.Name() << "," << spl1.Sex() << "," << spl1.Class() << "]" <<
//								"\t[" << spl2.Name() << "," << spl2.Sex() << "," << spl2.Class() << "]" <<
//								std::endl;
#endif

					}
					selector <<= 1;
				}
			}
			ms0p2 = rotr1(ms0p2);
			ms1p2 = rotr1(ms1p2);
		}
	}

	clog << "Precompute pairs: " << res.x << " alpha, " << res.y << " beta. " << tt.stop() << " sec." << endl;

	return res;
}


// Allocs and send to device all input data pertaining to calculation of ABK pairs.
template <typename T>
__host__ int allocSendData_device(PlinkReader<T> &pr, struct _dataPointersPairsD<T> &dptrs, const size_t nPairs, const struct _paramP1 &par)
{
	int nBytes = allocSendData_device(pr, dptrs.dpt, par);

	// TODO: Flags and copy data to a setup function.  Probably needs to change layout for coalescing
	unsigned char	*pFlag;
	CUDACHECK(cudaMalloc(&dptrs.pFlag,nPairs*sizeof(unsigned char)),par.gpuNum);
	CUDACHECK(cudaMallocHost(&pFlag,nPairs*sizeof(unsigned char)),par.gpuNum);
	nBytes += nPairs;

	uint2 cNumPairs = precompute_pairs(pr, pFlag);
	CUDACHECK(cudaMemcpy(dptrs.pFlag,pFlag,nPairs*sizeof(unsigned char),cudaMemcpyHostToDevice),par.gpuNum);

	CUDACHECK(cudaFreeHost(pFlag),par.gpuNum);

	return nBytes;
}

template <typename T>
__host__ void freeData_device(struct _dataPointersPairsD<T> &dptrs, const struct _paramP1 &par)
{
	freeData_device(dptrs.dpt, par);
	CUDACHECK(cudaFree(dptrs.pFlag),par.gpuNum);
}

#endif /* SETUP2_H_ */

/*
 * proc_gold_2.h
 *
 *  Created on: 30/12/2011
 *      Author: carlos
 *
 *  "Gold" standard routines for Alpha beta k generation
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

#ifndef PROC_GOLD_2_H_
#define PROC_GOLD_2_H_

#include "proc_gold.h"

/////////////////////
// Second Kernel.  Computation of per pair list
/////////////////////
template <typename T> void process_gold_2(PlinkReader<T> *pr, const size_t nPairs, const int nResPP, PairInteractionResult *ptops, const struct _paramP1 &par)
{
	_do_preload();
	const size_t nSNPs = pr->numSNPs();
	const size_t nELEs = pr->elementsSNP();
	const size_t nSamp = pr->numSamples();

	T *val0 = pr->classData(1)->mask();
	T *val1 = pr->classData(2)->mask();
	if (val0 == NULL || val1 == NULL)
	{
		cerr << "Error in class mask data" << std::endl;
		return;
	}

	////////////////////////////////////////////////////
	// Auxiliary setup
	////////////////////////////////////////////////////
	bool *isBetaPair = new bool[nPairs];
	std::memset(isBetaPair, 0, nPairs * sizeof(bool));
	bool *auxAPair = new bool[nPairs];
	bool *auxBPair = new bool[nPairs];
	cerr << "Aux vars: " << 3 * sizeof(bool) * nPairs << std::endl;

	Timer tt;
	std::memset(ptops, 0, nPairs * nResPP * sizeof(PairInteractionResult));
	cerr << "Zero " << tt.stop() << std::endl;

	double totTime = 0.0f;
	tt.start();

	size_t	nAlpha = 0;
	size_t	nBeta = 0;

	Timer 	t2;
	size_t  pairIndex = 0;
	// Gap in elements
	for (size_t gap = 0; gap < nELEs; gap++)
	{
		for (size_t k = 0; k < nELEs - gap; k++)
		{
			size_t e1 = k;
			size_t e2 = k + gap;
			size_t s1Idx = e1 * pr->SAMPLES_ELEMENT;
			size_t s2Idx = e2 * pr->SAMPLES_ELEMENT;
			// Valid and class masks
			T ms0p1 = val0[e1];	// Class 0 element 1
			T ms1p1 = val1[e1]; // Class 1 element 1
			T ms0p2 = val0[e2];	// Class 0 element 2
			T ms1p2 = val1[e2];	// Class 1 element 2
#ifdef DEBUG_PRINTS
//			fprintf(stderr, "%lu,%lu:(%lu,%lu) %lu eg:%lu\n", e1, e2, s1Idx, s2Idx, pairIndex, gap);
#endif

			for (size_t bitgap = 0; bitgap < pr->SAMPLES_ELEMENT; bitgap++)
			{
				T equal = (ms0p1 & ms0p2) | (ms1p1 & ms1p2);	// Bits on indicate pair is of equal class (beta)
				T diffe = (ms0p1 & ms1p2) | (ms1p1 & ms0p2);	// Bits on indicate pair is of different classes (alpha)
				T selector = (T) 0x1;
				if (e1 != e2 || bitgap > 0)
				{
					for (size_t n = 0; n < pr->SAMPLES_ELEMENT; n++)
					{
						size_t s1 = s1Idx + n;
						size_t s2 = s2Idx + (n + bitgap) % pr->SAMPLES_ELEMENT;
//						if (s1 < nSamp && s2 < nSamp && (e1 != e2 || n < pr->SAMPLES_ELEMENT - bitgap))
						if (s1 < nSamp && s2 < nSamp && s2 > s1)
						{
							pairIndex = INDEXATROWCOL_V00ND(s1,s2,nSamp);
							bool eqeq = ((equal & selector) == selector);
							bool dfdf = ((diffe & selector) == selector);
							bool vv = eqeq || dfdf;
							isBetaPair[pairIndex] = eqeq;
							if (vv)
							{
								if (eqeq)
									nBeta++;
								else
									nAlpha++;
							}
#ifdef DEBUG_PRINTS
//							Sample spl1 = pr->getSample(s1);
//							Sample spl2 = pr->getSample(s2);
//							cerr << "  " << pairIndex << "\t=\t" << s1 << "," << s2 << "\tv:" << vv << " e:" << eqeq << " d:" << dfdf <<
//									"\t[" << spl1.Name() << "," << spl1.Sex() << "," << spl1.Class() << "]" <<
//									"\t[" << spl2.Name() << "," << spl2.Sex() << "," << spl2.Class() << "]" <<
//									std::endl;
#endif
						}
						selector <<= 1;
					}
				}
				ms0p2 = rotr1(ms0p2);
				ms1p2 = rotr1(ms1p2);
			}

			double ti = tt.stop();
			tt.start();
			totTime += ti;
		}
	}
	fprintf(stderr,"AuxSetup: Total time: %g %g-%g/i.  numAlpha:%ld numBeta:%ld\n", t2.stop(), totTime, totTime/(double)nPairs, nAlpha, nBeta);

	//////////////////////////////////////////////////
	// SNP - SNP interaction computation
	//////////////////////////////////////////////////
	T *aMask0 = new T[nELEs];
	std::memset(aMask0, 0, nELEs * sizeof(T));
	T *aMask1 = new T[nELEs];
	std::memset(aMask1, 0, nELEs * sizeof(T));
	T *fBits = new T[nELEs];
	std::memset(fBits, 0, nELEs * sizeof(T));

#define		DEBUG_SNPS		4

	size_t stepCounter = 0;
	tt.start();
	for (size_t idxA = 0; idxA < min((int)nSNPs, DEBUG_SNPS); idxA++)
//	for (size_t idxA = 0; idxA < nSNPs; idxA++)
	{
		size_t rowOff = idxA * nELEs;
		T *snpA = pr->BitData() + rowOff;
		T *valA = pr->ValidData() + rowOff;
		count_type4 res = {0};
		for (size_t idxB = 0; idxB <= idxA; idxB++)
		{
			size_t colOff = idxB * nELEs;
			T *snpB = pr->BitData() + colOff;
			T *valB = pr->ValidData() + colOff;
			for (size_t m = 0; m < nELEs; m++)
			{
				aMask0[m] = val0[m] & valA[m] & valB[m];
				aMask1[m] = val1[m] & valA[m] & valB[m];
			}

//			for (int fun = 0; fun < NUM_FUNCS; ++fun)
			int fun = BFUNC_AND;
			{
				res = binFunc(snpA, snpB, aMask0, aMask1, fun, nELEs, fBits);
				float ee = entropyDelta4(res);
				size_t	covAlpha = 0;
				size_t	covBeta = 0;
				size_t	myAlpha = 0;
				size_t  myBeta = 0;
//				if (ee < 0.0f)
				{
					size_t  pairIndex = 0;
					// Gap in elements
					for (size_t gap = 0; gap < nELEs; gap++)
					{
						for (size_t k = 0; k < nELEs - gap; k++)
						{
							size_t e1 = k;
							size_t e2 = k + gap;
							size_t s1Idx = e1 * pr->SAMPLES_ELEMENT;
							size_t s2Idx = e2 * pr->SAMPLES_ELEMENT;

							T e1Bits = fBits[e1];
							T e2Bits = fBits[e2];
							T v1Bits = aMask0[e1] | aMask1[e1];
							T v2Bits = aMask0[e2] | aMask1[e2];
#ifdef DEBUG_PRINTS
							fprintf(stderr, "[%d,%d]\t%d,%d:\t%lx/%lx(%lx)\t%lx/%lx(%lx) Pi %d\n", idxA, idxB, e1, e2, e1Bits, v1Bits, aMask1[e1], e2Bits, v2Bits, aMask1[e2], pairIndex);
#endif

							for (size_t bitgap = 0; bitgap < pr->SAMPLES_ELEMENT; bitgap++)
							{
								// TODO: Revise equal policy, major & minor allele
								T equVal = (~(e1Bits ^ e2Bits));
								T valVal = v1Bits & v2Bits;										// Bits on if value is valid for sample
								T selector = (T) 0x1;
								if (e1 != e2 || bitgap > 0)
								{
									for (size_t n = 0; n < pr->SAMPLES_ELEMENT; n++)
									{
										size_t s1 = s1Idx + n;
										size_t s2 = s2Idx + (n + bitgap) % pr->SAMPLES_ELEMENT;

										if (s1 < nSamp && s2 < nSamp && s2 > s1)
										{
											pairIndex = INDEXATROWCOL_V00ND(s1,s2,nSamp);
											bool eqeq = ((equVal & selector) == selector);		// This pair has equal values
											bool vald = ((valVal & selector) == selector);		// This pair has valid values
											covAlpha += (vald && !isBetaPair[pairIndex] && !eqeq) ? 1 : 0;
											covBeta += (vald && isBetaPair[pairIndex] && eqeq) ? 1 : 0;
											// WARN:  myAlpha, myBeta are being used for debug !!!!!!!!!!!  Usage below is wrong
											myAlpha += (vald && !isBetaPair[pairIndex] && !eqeq) ? 1 : 0;
											myBeta += (vald && isBetaPair[pairIndex] && eqeq) ? 1 : 0;
//											myAlpha += (vald && !isBetaPair[pairIndex]) ? 1 : 0;
//											myBeta += (vald && isBetaPair[pairIndex]) ? 1 : 0;
											auxAPair[pairIndex] = vald && !isBetaPair[pairIndex] && !eqeq;
											auxBPair[pairIndex] = vald && isBetaPair[pairIndex] && eqeq;
#ifdef DEBUG_PRINTS
//											Sample spl1 = pr->getSample(s1);
//											Sample spl2 = pr->getSample(s2);
//											cerr << "  " << pairIndex << "\t=\t" << s1 << "," << s2 << "\tv:" << vv << " e:" << eqeq << " d:" << dfdf <<
//													"\t[" << spl1.Name() << "," << spl1.Sex() << "," << spl1.Class() << "]" <<
//													"\t[" << spl2.Name() << "," << spl2.Sex() << "," << spl2.Class() << "]" <<
//													std::endl;
#endif
										}
										selector <<= 1;
									}
								}
								e2Bits = rotr1(e2Bits);
								v2Bits = rotr1(v2Bits);
							}
							cerr << "\t" << e1 << "," << e2 << "\tA:" << myAlpha << "\tB:" << myBeta << std::endl;
							myAlpha = 0;
							myBeta = 0;
						}
					}

					cerr << "Interaction snpA " << idxA << " snpB " << idxB << " covAlpha " << covAlpha << " covBeta " << covBeta << std::endl;
					continue;

					// Compute global quality for this function
					// Could be covAlpha / myAlpha * ee     or     covAlpha / nAlpha * ee
					float Qalpha = (float) covAlpha * ee;
					// Could be covBeta / myNeta * ee     or     covBeta / nBeta * ee
					float Qbeta = (float) covBeta * ee;
					PairInteractionResult pirX;
					pirX.alphaC = covAlpha;
					pirX.betaC = covBeta;
					pirX.fun = fun;
					pirX.sA = idxA;
					pirX.sB = idxB;
					pirX.ent = 0.0f;
					for (size_t idxP = 0; idxP < nPairs; idxP++)
					{
						PairInteractionResult *LL = ptops + idxP * nResPP;
						// Store in LL->ent the worst so far
						if (auxAPair[idxP] && Qalpha < LL->ent)
						{
							float ww = Qalpha;
							int a2 = 0;
							for (int a1 = 1; a1 < nResPP; a1++)
								if (LL[a1].ent > ww)
								{
									a2 = a1;
									ww = LL[a1].ent;
								}
							if (a2 > 0)
								LL[0] = LL[a2];
							LL[a2] = pirX;
							LL[a2].ent = Qalpha;
//							if (idxP < 100)
//							{
//								cerr << "alpha " << idxP << " " << Qalpha << " " << a2 << endl;
//								for (int k = 0; k < nResPP; k++)
//									cerr << "  " << k << " " << LL[k].ent << endl;
//							}
						}
						else if (auxBPair[idxP] && Qbeta < LL->ent)
						{
							float ww = Qbeta;
							int a2 = 0;
							for (int a1 = 1; a1 < nResPP; a1++)
								if (LL[a1].ent > ww)
								{
									a2 = a1;
									ww = LL[a1].ent;
								}
							if (a2 > 0)
								LL[0] = LL[a2];
							LL[a2] = pirX;
							LL[a2].ent = Qbeta;
//							if (idxP < 100)
//							{
//								cerr << "beta " << idxP << " " << Qbeta << " " << a2 << endl;
//								for (int k = 0; k < nResPP; k++)
//									cerr << "  " << k << " " << LL[k].ent << endl;
//							}
						}
					}
				}
				stepCounter++;
				if (stepCounter % 10000 == 0)
				{
					double ti = tt.stop();
					totTime += ti;
					int validA = 0;
					int validB = 0;
					for (size_t p = 0; p < nELEs; p++)
					{
						validA += bitsElement(valA[p] & (val0[p] | val1[p]));
						validB += bitsElement(valB[p] & (val0[p] | val1[p]));
					}
					fprintf(stderr, "%ld,%ld fun:%d ALPHA:%ld/%ld/%ld BETA:%ld/%ld/%ld E:%g\n",
							idxA, idxB, fun, covAlpha, myAlpha, nAlpha, covBeta, myBeta, nBeta, ee);
					fprintf(stderr, "%lu: %d/%d, %d/%d.  vA:%d vB:%d. %gs %gms/step\n", stepCounter,
							res.w, res.y, res.x, res.z, validA, validB, ti, (totTime*1000.0f)/(double)stepCounter);
					tt.start();
				}
			}		// End of loop on functions
		}		// End of loop on SNP B
	}		// End of loop on SNP A
	totTime += tt.stop();
	clog << "process_gold_2: " << totTime << " " << std::endl;

	delete[] isBetaPair;
	delete[] auxAPair;
	delete[] auxBPair;
	delete[] fBits;
	delete[] aMask0;
	delete[] aMask1;
}

#endif /* PROC_GOLD_2_H_ */

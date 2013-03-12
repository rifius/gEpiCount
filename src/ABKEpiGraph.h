/*
 * ABKEpiGraph.h
 *
 *  Created on: 10/03/2013
 *      Author: carlos
 */

#ifndef ABKEPIGRAPH_H_
#define ABKEPIGRAPH_H_

#include <new>
#include <map>
#include <list>
#include <vector>
#include "../inc/gABKEpi.h"
#include "reader/PlinkReader.h"

template <class Key, typename T>
class ABKEpiGraph {
public:
	static const char	P_VALID = 0x01;
	static const char	P_ALPHA = 0x02;
private:
	map<Key, TrackedPairInteractionResult>	pirMap;	// Private map to keep Interaction Results
	vector<list<Key> >	pairList;		// Array of lists for each pair
	int			_nPairs;
	int			_nResPP;
	int			*wCovers;		// Convenience array of worst covers
	unsigned char *pFlag;		// ALPHA or BETA or VALID
protected:
	ABKEpiGraph();
public:
	ABKEpiGraph(int nPairs, int nResPP);
	virtual ~ABKEpiGraph();

	uint2 precomputePairs(const PlinkReader<T> &pr);

	int getWorstAlphaCover();
	int getWorstBetaCover();
	const int *getWorstCovers();
	const unsigned char *getPairFlags() const;

	int nPairs() const
	{
		return _nPairs;
	}

	int nResPP() const
	{
		return _nResPP;
	}
};

template <class Key, typename T>
ABKEpiGraph<Key, T>::ABKEpiGraph(int nPairs, int nResPP)
{
	this->_nPairs = nPairs;
	this->_nResPP = nResPP;
	this->pairList = vector<list<Key> >(nPairs);
	this->pirMap = std::map<Key, TrackedPairInteractionResult>();
	this->wCovers = new int [nPairs];
	this->pFlag = new unsigned char [nPairs];
}

template <class Key, typename T>
ABKEpiGraph<Key, T>::~ABKEpiGraph()
{
//	typename std::map<Key, TrackedPairInteractionResult>::iterator x;
//	for(x = this->pirMap.begin(); x != this->pirMap.end(); x++)
//	{
//		delete x->second;
//	}
	this->pirMap.clear();
//	delete this->pirMap;
	for(int j = 0; j < this->_nPairs; j++)
	{
		this->pairList[j].clear();
	}
//	delete this->pairList;
	delete[] this->wCovers;
	delete[] this->pFlag;
}

template<class Key, typename T>
inline int ABKEpiGraph<Key, T>::getWorstAlphaCover()
{
	return -1;
}

template<class Key, typename T>
inline int ABKEpiGraph<Key, T>::getWorstBetaCover()
{
	return -1;
}

template<class Key, typename T>
inline const int* ABKEpiGraph<Key, T>::getWorstCovers()
{
	return this->wCovers;
}

template<class Key, typename T>
inline const unsigned char* ABKEpiGraph<Key, T>::getPairFlags() const
{
	return this->pFlag;
}

template<class Key, typename T>
uint2 ABKEpiGraph<Key, T>::precomputePairs(const PlinkReader<T> &pr)
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
							this->pFlag[pairIndex] = P_VALID;
							if (eqeq)
								res.y++;
							else	// is alpha
							{
								this->pFlag[pairIndex] |= P_ALPHA;
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






#endif /* ABKEPIGRAPH_H_ */

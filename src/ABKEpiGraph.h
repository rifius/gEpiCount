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
#include <exception>
#include <typeinfo>
#include <iostream>
#include "../inc/gABKEpi.h"
#include "reader/PlinkReader.h"

class pirException : std::exception
{
	const string msg;
public:
	pirException(const string what) : msg(what) {};
	~pirException() throw() {};
	const char *what() const throw () { return msg.c_str(); };
};

template <class Key, typename T>
class ABKEpiGraph {
public:
	static const char	P_VALID = 0x01;
	static const char	P_ALPHA = 0x02;
private:
	std::map<Key, TrackedPairInteractionResult>	pirMap;	// Private map to keep Interaction Results
	std::vector<std::list<Key> >	pairList;		// Array of lists for each pair
	int			_nPairs;
	int			_nResPP;
	int			*wCovers;		// Convenience array of worst covers
	int 		_worstAC;
	int			_worstBC;
	unsigned char *pFlag;		// ALPHA or BETA or VALID
	bool		dirty;			// worstCovers must be recomputed
	const Key	keyForPIR(const PairInteractionResult &pir);
	static const int SHIFTIDX = (sizeof(Key)*8+4)/2;
protected:
	ABKEpiGraph();
public:
	ABKEpiGraph(int nPairs, int nResPP);
	virtual ~ABKEpiGraph();

	uint2 precomputePairs(const PlinkReader<T> &pr);
	int updatePair(int pair, const PairInteractionResult *irs, int pirLen,
			const PILindex_t *idxs,	int maxIdxLen, PILindex_t emptyIndex = EMPTY_INDEX_B2);

	int getWorstAlphaCover();
	int getWorstBetaCover();
	const int *getWorstCovers();
	const unsigned char *getPairFlags() const;

	void print();

	int nPairs() const
	{
		return _nPairs;
	}

	int nResPP() const
	{
		return _nResPP;
	}

	bool isAlpha(int pair)
	{
		unsigned char f = this->pFlag[pair];
		return ((f & ABKEpiGraph::P_VALID) == ABKEpiGraph::P_VALID) && ((f & ABKEpiGraph::P_ALPHA) == ABKEpiGraph::P_ALPHA);
	}
	bool isBeta(int pair)
	{
		unsigned char f = this->pFlag[pair];
		return ((f & ABKEpiGraph::P_VALID) == ABKEpiGraph::P_VALID) && ((f & ABKEpiGraph::P_ALPHA) == 0);
	}
	bool isValid(int pair)
	{
		unsigned char f = this->pFlag[pair];
		return ((f & ABKEpiGraph::P_VALID) == ABKEpiGraph::P_VALID);
	}
};

template <class Key, typename T>
ABKEpiGraph<Key, T>::ABKEpiGraph(int nPairs, int nResPP)
{
	if (sizeof(Key) < 4)	// Arbitrary minimum usable key
	{
		throw new pirException(string("Key is too small: ") + typeid(Key).name());
	}
	this->_nPairs = nPairs;
	this->_nResPP = nResPP;
	this->dirty = false;
	this->_worstAC = 0;
	this->_worstBC = 0;
	this->pairList = std::vector<std::list<Key> >(nPairs);
	this->pirMap = std::map<Key, TrackedPairInteractionResult>();
	this->wCovers = new int [nPairs];
	this->pFlag = new unsigned char [nPairs];
	std::memset(this->wCovers, 0, nPairs * sizeof(int));
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

	for(int j = 0; j < this->_nPairs; j++)
	{
		this->pairList[j].clear();
	}

	delete[] this->wCovers;
	delete[] this->pFlag;
}

template<class Key, typename T>
inline int ABKEpiGraph<Key, T>::getWorstAlphaCover()
{
	return this->_worstAC;
}

template<class Key, typename T>
inline int ABKEpiGraph<Key, T>::getWorstBetaCover()
{
	return this->_worstBC;
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
//	std::cerr << "  pairIndex\t=\ts1,s2 [e1,e2]\tv:valid f:flags\t[s1.Name,s1.Sex,s1.Class]\t[s2.Name,s2.Sex,s2.Class]" << std::endl;
#endif

	Timer tt;
	int nEP = (pr.elementsSNP() * (pr.elementsSNP() + 1)) / 2;
	for (int epi = 0; epi < nEP; epi++)
	{
		int e1, e2;
		ROWCOLATINDEX_V0D(epi,pr.elementsSNP(),e1,e2);

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
//						std::cerr << "  " << pairIndex << "\t=\t" << s1 << "," << s2 << " [" << e1 << "," << e2 << "]" <<
//									 "\tv:" << vv << " f:" << (int) flags[pairIndex] <<
//									 "\t[" << spl1.Name() << "," << spl1.Sex() << "," << spl1.Class() << "]" <<
//									 "\t[" << spl2.Name() << "," << spl2.Sex() << "," << spl2.Class() << "]" <<
//									 std::endl;
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


/* Method to update the contents for a particular pair with the list of interaction results
 * Receives as arguments:
 * pair			Pair index to update
 * irs[pirLen]	Array of PairInteractionResult referenced by this pair updates
 * idxs[maxIdxLen]	Array of PILindex_t indices to the irs[] array
 * emptyIndex	Special index value indicating an non-valid index in idxs
 *
 * The array idxs may not be completely filled with valid references to irs[],
 * scanning stops at the first out-of-range or emptyIndex value (first case with message).
 * This method updates the worstCover value for the pair, maintains the refcount for stored
 * PairInteractionResult in the map, and keep the size of the list for the pair up to
 * nResPP.  The class expects that values provided as PairInteractionResult in irs[] are
 * consistent with previous invocations for different pairs.  An error is issued.
 * TODO: Make some errors exceptions and be more lenient.
 */
template<class Key, typename T>
inline int ABKEpiGraph<Key, T>::updatePair(int pair, const PairInteractionResult* irs, int pirLen,
		const PILindex_t* idxs, int maxIdxLen, PILindex_t emptyIndex)
{
	static const string _binaryFuncName[BFUNC_LAST + 1] = { "AND", "NAND", "ANDN", "NANDN", "XOR", "Invalid" };
	int newPirs = 0;
	for (int k = 0; k < maxIdxLen && idxs[k] != emptyIndex; k++)
	{
		PILindex_t cIndex = idxs[k];
		if (cIndex >= pirLen)
		{
			std::cerr << "Error:" << typeid(this).name() << ": Index beyond supplied array " << cIndex << " >= " << pirLen << std::endl;
			break;
		}
		Key aKey = keyForPIR(irs[cIndex]);
		typename std::map<Key, TrackedPairInteractionResult>::iterator mit = this->pirMap.find(aKey);
		if (mit == this->pirMap.end())
		{
			// Key does not exist, add the element to the map
			TrackedPairInteractionResult mtpi = {irs[cIndex], 0};
			mit = (this->pirMap.insert(std::pair<Key,TrackedPairInteractionResult>(aKey,mtpi))).first;
//			mit = this->pirMap.find(aKey);
			newPirs++;
		}
		else
		{
			// The key is there, check values.
			const TrackedPairInteractionResult *p = &(mit->second);
			if (p->ir.alphaC != irs[cIndex].alphaC || p->ir.betaC != irs[cIndex].betaC)
			{
				std::cerr << "Error:" << typeid(this).name() << ": Stored element different from provided." << std::endl <<
						"\tInteraction: " << irs[cIndex].sA << " " << _binaryFuncName[irs[cIndex].fun] << " " << irs[cIndex].sB <<
						"\tcA:" << irs[cIndex].alphaC << "(n)," << p->ir.alphaC << "(s)" <<
						"\tcB:" << irs[cIndex].betaC << "(n)," << p->ir.betaC <<	"(s)\trefCount:" << p->refCount << std::endl;
				continue;
			}
		}
		TrackedPairInteractionResult *tpi = &(mit->second);
		typename std::list<Key>::iterator lit;
		int nCover = 0;
		if (this->isAlpha(pair))
		{
			nCover = irs[cIndex].alphaC;
			for (lit = this->pairList[pair].begin(); lit != this->pairList[pair].end() && nCover < this->pirMap[*lit].ir.alphaC; ++lit) ;
		}
		else if (this->isBeta(pair))
		{
			nCover = irs[cIndex].betaC;
			for (lit = this->pairList[pair].begin(); lit != this->pairList[pair].end() && nCover < this->pirMap[*lit].ir.betaC; ++lit) ;
		}
		if (this->pairList[pair].size() < this->_nResPP || lit != this->pairList[pair].end())
		{
			this->pairList[pair].insert(lit,aKey);
			tpi->refCount++;
		}
		else
		{
			// Somthing is rotten in Denmark
		}
	}
	return newPirs;
}

/*
 * Private method to generate keys.  Primitive.
 * Keys will encode both indexes and function.
 * Selection of method is based on Key size.
 * We need 3 bits to encode the function (in 0-5), and the rest is divided in
 * two equal parts for each SNP index.
 * Throws an exception when index exceeds the alloted bits.
 */
template<class Key, typename T>
inline const Key ABKEpiGraph<Key, T>::keyForPIR(const PairInteractionResult& pir)
{
	int idxA = pir.sA;
	int idxB = pir.sB;
	int fun = pir.fun;
	// Normalisation
	if (pir.sA > pir.sB)
	{
		idxA = pir.sB;
		idxB = pir.sA;
		switch(pir.fun)
		{
		case BFUNC_ANDN:	fun = BFUNC_NAND;	break;
		case BFUNC_NAND:	fun = BFUNC_ANDN;	break;
		}
	}
	Key key = (((Key) idxA) << SHIFTIDX) | (((Key) idxB) << 4) | (Key)(fun & 0x0007);
	return key;
#ifdef DEBUG_PRINTS
#endif
}

template<class Key, typename T>
void ABKEpiGraph<Key, T>::print()
{
	static const string _binaryFuncName[BFUNC_LAST + 1] = { "AND", "NAND", "ANDN", "NANDN", "XOR", "Invalid" };
	typename std::map<Key, TrackedPairInteractionResult>::iterator mit;
	for (mit = this->pirMap.begin(); mit != this->pirMap.end(); mit++)
	{
		Key	key = mit->first;
		TrackedPairInteractionResult *tpi = &(mit->second);
		std::clog << "Key:" << key << "\trefCount:" << tpi->refCount << "\n" <<
				"\tInteraction: " << tpi->ir.sA << " " << _binaryFuncName[tpi->ir.fun] << " " << tpi->ir.sB <<
				"\tcA:" << tpi->ir.alphaC << "\tcB:" << tpi->ir.betaC << std::endl;
	}
}

#endif /* ABKEPIGRAPH_H_ */

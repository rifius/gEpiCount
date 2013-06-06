/*
 * PlinkReader.h
 *
 *  Created on: 17/12/2011
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

#ifndef PLINKREADER_H_
#define PLINKREADER_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cstring>
#include <omp.h>

#include "SampleClassInfo.h"
#include "../misc/bitops.h"

using namespace std;

// Class to hold static information about the SNP
class SNP
{
	string snpId;
	char allele;
	int chr, pos;
public:
	SNP(const string snpId, char allele, int chr, int pos)
	{
		this->snpId = snpId;
		this->allele = allele;
		this->chr = chr;
		this->pos = pos;
	}
	virtual ~SNP() {};

	int Chr() const
	{
		return chr;
	}
	int Pos() const
	{
		return pos;
	}
	string Id() const
	{
		return snpId;
	}
	string Alleles() const
	{
		return string(1, allele);
	}
};

// Class to hold static information about the Sample
class Sample
{
	string name;
	int sclass, sex;
public:
	static const int INVALID = -9;
	enum
	{
		MALE, FEMALE, UNKNOWN
	};

	Sample(const string name, int isex, int aclass)
	{
		this->name = name;
		this->sclass = aclass;
		switch (isex)
		{
		case 1: 	this->sex = MALE;	break;
		case 2:		this->sex = FEMALE;	break;
		default:	this->sex = UNKNOWN;break; // TODO: check condition for invalid
		}
	}
	virtual ~Sample() {};

	string Name() const
	{
		return name;
	}
	int Class() const
	{
		return sclass;
	}
	int Sex() const
	{
		return sex;
	}
};


// Template class for the PlinkReader.  The template parameter is the type to be used as binary
// data store, must be an unsigned integer type.
template <class ET> class PlinkReader
{
public:
	static const unsigned int ELEMENT_SIZE = sizeof(ET);
	static const unsigned int SAMPLES_ELEMENT = ELEMENT_SIZE * 8;
	friend class SampleClassInfo<ET>;

protected:
	static const ET BIT_ON = ((ET) 0x1) << (SAMPLES_ELEMENT - 1);
	vector<Sample> samples;
	vector<SNP> snps;
	map<int,SampleClassInfo<ET>*> sampleClasses;
	string basepath;
	// Checks .fam & .bim files exist
	virtual bool checkBaseConsistency(void)
	{
		// We expect to have a .fam and a .bim file
		bool ok = false;
		ok = this->readSamples();
		this->baseChecked = ok && this->readSNPs();
		this->elemSNP = ((this->numSamples() + (SAMPLES_ELEMENT - 1)) / SAMPLES_ELEMENT);
		this->classifySamples();
		return this->baseChecked;
	}

	// Prepare space for  data
	void setDataVars(void)
	{
		size_t totElements = this->numSNPs() * this->elementsSNP();
		this->vData = new ET[totElements];
		this->bData = new ET[totElements];
		memset(this->vData, 0, totElements * sizeof(ET));
		memset(this->bData, 0, totElements * sizeof(ET));
	}


public:
	PlinkReader(const string basePath)
	{
		this->basepath = basePath;
		this->elemSNP = 0;
		this->bData = NULL;
		this->vData = NULL;
		this->baseChecked = false;
	}
	virtual ~PlinkReader()
	{
		if (this->bData != NULL)
			delete[] this->bData;
		if (this->vData != NULL)
			delete[] this->vData;

		typename std::map<int,SampleClassInfo<ET>* >::iterator ite;
		for(ite = this->sampleClasses.begin(); ite != this->sampleClasses.end(); ite++)
			delete ite->second;
		this->sampleClasses.clear();
	}

	// Read the data and fill vectors and bit data matrices
	virtual bool readData(void) = 0;
	// Check files exist, read samples and snps and check data file size
	virtual bool checkConsistency(void) = 0;

	vector<Sample> getSamples() const
	{
		return this->samples;
	}

	vector<SNP> getSnps() const
	{
		return this->snps;
	}

	Sample getSample(int index) const
	{
		return this->samples.at(index);
	}

	SNP getSnp(int index) const
	{
		return this->snps.at(index);
	}

	int numSamples(void) const
	{
		return this->samples.size();
	}

	int numSNPs(void) const
	{
		return this->snps.size();
	}

	int elementsSNP(void) const
	{
		return this->elemSNP;
	}

	virtual size_t DataSize(void)
	{
		return ELEMENT_SIZE * this->elementsSNP() * this->numSNPs();
	}
	virtual ET *BitData(void) const
	{
		return this->bData;
	}
	virtual ET *ValidData(void) const
	{
		return this->vData;
	}
	SampleClassInfo<ET> *classData(int key) const
	{
		if (this->sampleClasses.count(key) == 0)
			return NULL;
		else
			return (*(this->sampleClasses.find(key))).second;
	}
	// Check quality of loaded data
	// SNPs with more than maxMissSamples missing values ratio are reported (default 5%)
	// Samples with more than maxMissSNPs missing SNPs ratio are reported (default 5%)
	// If remove = true, then non-conforming data is removed [NOT IMPLEMENTED]
	virtual bool checkQuality(const bool remove = false, const float maxMissSNPs = 0.05f, const float maxMissSamples=0.05f)
	{
		if (!this->baseChecked || this->vData == NULL)
		{
			cerr << "Data not loaded" << std::endl;
			return false;
		}
		_do_preload();
//#define NO_MT
#ifndef NO_MT
		int numThreads = omp_get_max_threads();
#else
		int numThreads = 1;
#endif
		const int nSNPs = this->numSNPs();
		const int nSamps = this->numSamples();

		int ncSamp = 0;
		int chunk = nSamps / numThreads;
		clog << "Checking data in " << numThreads << " chunks of " << chunk << " Samples." << std::endl;
#ifndef NO_MT
#pragma omp parallel num_threads(numThreads) shared(numThreads, chunk, stderr) reduction(+:ncSamp)
#endif
		{
#ifndef NO_MT
			int tid = omp_get_thread_num();
#else
			int tid = 0;
#endif
			int start = tid * chunk;
			int end = (tid == (numThreads-1) ? nSamps: start + chunk);
			for (int j = start; j < end; j++)
			{
				int nb = j % this->SAMPLES_ELEMENT;
				int ele = j / this->SAMPLES_ELEMENT;
				int valid = 0;
				ET sele = (ET) 0x1 << nb;
				for (int k = 0; k < nSNPs; k++)
				{
					int idx = k * this->elemSNP + ele;
					ET data = this->vData[idx];
					if (sele == (sele & data))
						valid++;
				}
				float sMiss = (float) (nSNPs - valid) / (float) nSNPs;
				if (sMiss > maxMissSamples)
				{
#ifndef NO_MT
#pragma omp critical
#endif
					fprintf(stderr, "Sample %d:\t%s(%d)\tnMiss:%d\t(%4g)\n", j, this->getSample(j).Name().c_str(),
							this->getSample(j).Class(), ((int) nSNPs - valid), sMiss);
					ncSamp++;
				}
				if (tid == 0 && (j % 100) == 0)
					fprintf(stderr, "%d %d %g\n", j, (int) (nSNPs - valid), sMiss);
			}
		}
		fprintf(stderr, "\n%d Samples checked, %d with more than %4g missing values\n\n", this->numSamples(), ncSamp, maxMissSamples);

//#define NO_MT
		int ncSNPs = 0;
		chunk = nSNPs / numThreads;
		clog << "Checking data in " << numThreads << " chunks of " << chunk << " SNPs." << std::endl;
#ifndef NO_MT
#pragma omp parallel num_threads(numThreads) shared(numThreads, chunk, stderr, bits_in_i2) reduction(+:ncSNPs)
#endif
		{
#ifndef NO_MT
			int tid = omp_get_thread_num();
#else
			int tid = 0;
#endif
			int start = tid * chunk;
			int end = (tid == (numThreads-1) ? nSNPs: start + chunk);
			for (int j = start; j < end; j++)
			{
				ET *vd = this->vData + j * this->elemSNP;
				int valid = 0;
				for (int k = 0; k < this->elemSNP; k++)
					valid += bitsElement(vd[k]);
				float sMiss = (float) (nSamps - valid) / (float) nSamps;
				if (sMiss > maxMissSNPs)
				{
#ifndef NO_MT
#pragma omp critical
#endif
					fprintf(stderr, "SNP %d:\t%s\tnMiss:%d\t(%4g)\n", j, this->getSnp(j).Id().c_str(), ((int) nSamps - valid), sMiss);
					ncSNPs++;
				}
				if (tid == 0 && (j % 10000) == 0)
					fprintf(stderr, "%d %d %g\n", j, (int) (nSamps - valid), sMiss);
			}
		}
		fprintf(stderr, "\n%d SNPs checked, %d with more than %4g missing values\n\n", this->numSNPs(), ncSNPs, maxMissSNPs);
#undef NO_MT

		return ncSNPs == 0 && ncSamp == 0;
	}

private:
	ET *bData; // Bit Data "Major" (First) allele
	ET *vData; // Valid Data Mask
	ET **vClassMask; // Array of Valid samples per class
	bool baseChecked;
	int elemSNP; // Elements per SNP

	// Reads a .fam file and loads samples
	bool readSamples(void)
	{
		ifstream is;
		stringstream sis(stringstream::in | stringstream::out);
		filebuf * fb = is.rdbuf();
		string fname = this->basepath + ".fam";
		fb->open(fname.c_str(), ios::in);
		if (!fb->is_open())
			return false;
		int linecount = 0;
		while (!is.eof())
		{
			string sn, line;
			int d, e;
			size_t n;
			getline(is, line);
			if (line.size() < 5)
				continue;
			n = line.find_first_of(" \t", 0);
			sn = line.substr(0, n);
			sscanf(line.c_str(), "%*s %*s %*d %*d %d %d", &d, &e);
			Sample s(sn, d, e);
			samples.push_back(s);
			linecount++;
		}
		fb->close();
		clog << fname << ": " << linecount << " lines read, " << samples.size() << " samples" << endl;
		return true;
	}

	// Reads a .bim file and loads snps
	bool readSNPs(void)
	{
		ifstream is;
		stringstream sis(stringstream::in | stringstream::out);
		filebuf * fb = is.rdbuf();
		string fname = this->basepath + ".bim";
		fb->open(fname.c_str(), ios::in);
		if (!fb->is_open())
			return false;
		int linecount = 0;
		while (!is.eof())
		{
			string sn, line;
			int a, b, c;
			char d, e;
			getline(is, line);
			if (line.size() < 9)
				continue;
			sis.str(line);
			sis >> a >> sn >> b >> c >> d >> e;
			SNP s(sn, d, a, c);
			snps.push_back(s);
			SNP t(sn, e, a, c);
			snps.push_back(t);
			linecount++;
		}
		fb->close();
		clog << fname << ": " << linecount << " lines read, " << snps.size() << " SNPs" << endl;
		return true;
	}

	// Classifies the samples and constructs masks per class
	bool classifySamples(void)
	{
		ET msk;

		for (int j = 0; j < this->numSamples(); j++)
		{
			int eln = j / SAMPLES_ELEMENT;
			int els = j % SAMPLES_ELEMENT;
			if (this->samples[j].Class() != Sample::INVALID)
				msk = ((ET) 0x1) << els;
			else
				msk = 0;
			int cc = this->samples[j].Class();
			if (this->sampleClasses.count(cc) == 0)
			{
				SampleClassInfo<ET> *sci = new SampleClassInfo<ET>(cc, (int) this->numSamples());
				sci->mask()[eln] = msk;
				this->sampleClasses.insert(pair<int,SampleClassInfo<ET>*>(cc,sci));
			}
			else
			{
				(this->sampleClasses.find(cc))->second->incrementCounter();
				(this->sampleClasses.find(cc))->second->mask()[eln] |= msk;
			}
		}
		typename std::map<int,SampleClassInfo<ET>* >::iterator it;
		for (it = this->sampleClasses.begin(); it != this->sampleClasses.end(); it++)
		{
			SampleClassInfo<ET> *sci = it->second;
			clog << "Class " << sci->classId() << " " << sci->mumSamplesClass() << " samples" << std::endl;
		}
		return true;
	}

};

#endif /* PLINKREADER_H_ */

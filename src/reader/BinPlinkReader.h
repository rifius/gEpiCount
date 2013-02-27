/*
 * BinPlinkReader.h
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

#ifndef BINPLINKREADER_H_
#define BINPLINKREADER_H_

#include <fcntl.h>
#include <sys/mman.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "PlinkReader.h"
#include "../misc/Timer.h"

template <class ET> class BinPlinkReader: public PlinkReader<ET>
{
private:
	bool checked;
	bool SNPMajor;

public:
	BinPlinkReader(const string basePathName) : PlinkReader<ET>(basePathName)
	{
		this->checked = false;
		this->SNPMajor = false;
	};
	virtual ~BinPlinkReader() {};

	bool isSNPMajor(void)
	{
		return this->SNPMajor;
	}

	// Check files exist, read samples and snps and check data file size
	bool checkConsistency(void)
	{
		// We expect to have a .bim, a .fam and a .bed
		ifstream is;
		filebuf * fb = is.rdbuf();
		bool ok;
		ok = this->checkBaseConsistency();
		if (!ok)
			return ok;
		string fname = this->basepath + ".bed";
		fb->open(fname.c_str(), ios::in);
		if (!fb->is_open())
			return false;
		int expect = (this->numSNPs()/2) * ((this->numSamples() + 3) / 4) + 3;

		unsigned short magic;
		unsigned char order;
		is.read((char*)&magic, sizeof(magic));
		is.read((char*)&order, 1);

		is.seekg(0, ios_base::end);
		int size = is.tellg();
		fb->close();
		if (magic != 0x1b6c)
		{
			clog << fname << ": Incorrect Magic " << hex << magic << dec << endl;
			return false;
		}
		else if (expect != size)
		{
			clog << fname << ": Size (" << size << ") different than expected (" << expect << ")" << endl;
			return false;
		}
		else
		{
			this->SNPMajor = (order == 0x1);
			clog << fname << ": " << (this->SNPMajor ? "SNP" : "Sample") << " major order, size ok." << endl;
		}
		this->checked = true;
		return ok;
	}

	// Read the data and fill vectors and bit data matrices
	bool readData(void)
	{
		if (!this->checked && !this->checkConsistency())
			return false;
		// We do the read via memory map
		int fd;
		unsigned char *data;
		string fname = this->basepath + ".bed";
		fd = open(fname.c_str(), O_RDONLY);
		if (fd == -1)
			return false;

		this->setDataVars();
		// Map the data. This is for SNP major order
		size_t expect = (this->numSNPs()/2) * ((this->numSamples() + 3) / 4) + 3;
		data = (unsigned char *) mmap(0, expect, PROT_READ, MAP_SHARED, fd, 0);
		if (data == MAP_FAILED) {
			perror("Error mapping data");
			close(fd);
			return false;
		}

		Timer tt;
		// Read the data.
		int snpDataBytes = ((this->numSamples() + 3) / 4); 	// Bytes per SNP
#ifndef NO_MT
		int numThreads = omp_get_max_threads();
#else
		int numThreads = 1;
#endif
		int chunk = this->numSNPs() / numThreads;
		chunk = (chunk >> 1) << 1;	// We need an even number
		size_t *sMiss = new size_t[numThreads * this->numSamples()];
		std::memset(sMiss, 0, numThreads * this->numSamples() * sizeof(size_t));
		clog << "Reading data in " << numThreads << " chunks of " << chunk << " SNPs. " << snpDataBytes << "b/SNP, " << this->elementsSNP() << "e/SNP" << endl;
#ifndef NO_MT
#pragma omp parallel num_threads(numThreads) shared(data, numThreads, chunk, clog, snpDataBytes, sMiss)
#endif
		{
			unsigned char dd = 0;
			int mp, j, me = 0;
#ifndef NO_MT
			int tid = omp_get_thread_num();
#else
			int tid = 0;
#endif
			int start = (tid * chunk);
			int end = (tid == (numThreads-1) ? this->numSNPs(): start + chunk);
			int doffset = start / 2 * snpDataBytes + 3;			// Offset in map
			int belement = start * this->elementsSNP();
			// Debug
//#pragma omp critical
//			fprintf(stderr, "tid %d, start %d, end %d, doff %d, belement %d\n", tid, start, end, doffset, belement);
			start /= 2;
			end /= 2;
			ET vMsk = 0;
			ET bDat_A = 0;
			ET bDat_a = 0;

			for (j = start; j < end; j++)
			{
				int pMiss = 0;
				for (int k = 0; k < this->numSamples(); k++)
				{
					mp = k % 4;
					me = (k + 1) % PlinkReader<ET>::SAMPLES_ELEMENT;
					if (mp == 0) {
						dd = data[doffset];
						doffset++;
					}
					int ss = dd & 0x3;
					dd >>= 2;
					// Interpret data according to http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml
					switch(ss) {
					case 0:		// Homozygous in major
						bDat_A 	|= PlinkReader<ET>::BIT_ON;
						vMsk 	|= PlinkReader<ET>::BIT_ON;
						break;
					case 2: 	// Heterozygous
						bDat_A 	|= PlinkReader<ET>::BIT_ON;
						bDat_a 	|= PlinkReader<ET>::BIT_ON;
						vMsk 	|= PlinkReader<ET>::BIT_ON;
						break;
					case 3:		// Homozygous in minor
						bDat_a 	|= PlinkReader<ET>::BIT_ON;
						vMsk 	|= PlinkReader<ET>::BIT_ON;
						break;
					case 1:		// Missing value
						sMiss[tid * this->numSamples() + k]++;
						pMiss++;
						break;	// Missing genotype
					default:
						cerr << "We should't be here ! " << ss << std::endl;
						break;
					}
					if (me == 0)
					{
						this->BitData()[belement] = bDat_A;
						this->ValidData()[belement] = vMsk;
						this->BitData()[belement+this->elementsSNP()] = bDat_a;
						this->ValidData()[belement+this->elementsSNP()] = vMsk;
						bDat_A = bDat_a = vMsk = 0;
						belement++;
					}
					else
					{
						bDat_A	>>= 1;
						bDat_a 	>>= 1;
						vMsk	>>= 1;
					}
				}
				if (me != 0)
				{
					bDat_A >>= (PlinkReader<ET>::SAMPLES_ELEMENT - me - 1);
					bDat_a >>= (PlinkReader<ET>::SAMPLES_ELEMENT - me - 1);
					vMsk >>= (PlinkReader<ET>::SAMPLES_ELEMENT - me - 1);
					this->BitData()[belement] = bDat_A;
					this->ValidData()[belement] = vMsk;
					this->BitData()[belement+this->elementsSNP()] = bDat_a;
					this->ValidData()[belement+this->elementsSNP()] = vMsk;
					bDat_A = bDat_a = vMsk = 0;
					belement++;
				}
				belement += this->elementsSNP();
			}
			// Debug
//#pragma omp critical
//			fprintf(stderr, "tid %d, belement %d\n", tid, belement);
		}	// End parallel

		int tot = 0;
		for (int k = 0; k < this->numSamples(); k++)
		{
			for (int j = 0; j < numThreads; j++)
				tot += sMiss[j * this->numSamples() + k];
		}
		delete[] sMiss;

		double totCallRate = 1.0 - (double) tot / ((double) this->numSNPs() * (double) this->numSamples());
		clog << "Process data: " << tt.stop() << "s. total missing " << tot << ", total call rate " << totCallRate << endl;
		munmap(data, expect);
		close(fd);
		return true;
	}

};

#endif /* BINPLINKREADER_H_ */

/*
 ============================================================================

 Name        : gEpiCount.c
 Author      : Carlos Riveros
 Version     :
 Copyright   : (C) Carlos Riveros, 2012
 Description : gEpiCount Boolean entropy-based epistasis
 ============================================================================
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
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../inc/gEpiCount.h"
#include "../inc/gABKEpi.h"
#include "proc_gold_1.h"
#include "proc_gold_2.h"
#include "proc_gpu.h"

#define		DEFAULT_TOPMOST		1000000
#define		DEFAULT_GPUNUM		0
#define		DEFAULT_BLOCK_SIDE	16
#define		DEFAULT_MAXRESULTS	16		// In units of ONEMEGA
#define		DEFAULT_MISS_SNPS   0.05
#define		DEFAULT_MISS_SAMP   0.05

#define		PROGNAME		"gEpiCount"

static void printHelp(void);
static bool getParams(int argc, char *argv[], struct _paramP1 &opar);

//#define UI4ELE
int main (int argc, char *argv[]) {

	struct _paramP1 par;

	clog << PROGNAME << ", version: " << progVersion << endl << endl;

	if (!getParams(argc, argv, par))
		exit(3);

	size_t	nSAMP;
#ifdef		UI4ELE
	PlinkReader<ui4> *plkr = new BinPlinkReader<ui4>(par.name);
#else
	PlinkReader<ui8> *plkr = new BinPlinkReader<ui8>(par.name);
#endif
	if (!plkr->checkConsistency())
	{
		printf("Error in %s\n", par.name.c_str());
	}
	plkr->readData();

	printf("\nElement Size: %d\n", plkr->ELEMENT_SIZE);
	printf("Size of each Data Matrix: %ld (%ldMB)\n", plkr->DataSize(), (plkr->DataSize()+ONEMEGA-1)/ONEMEGA);
//	for (int k = 0; k < plkr->numSNPs() && k < 1000; k += 10)
//	{
//		print_data(plkr, k, 2);
//	}
	plkr->checkQuality(par.qaRemoveMissing, par.qaMissSNPs, par.qaMissSamp);
	nSAMP = plkr->numSamples();

	printf("Size of array of %d topmost interaction results: %ld (%ldMB) (+1x buffer)\n", par.topmost, 2 * par.topmost * sizeof(InteractionResult),
			(2 * par.topmost * sizeof(InteractionResult)+ONEMEGA-1)/ONEMEGA);
	// For CPU, we use the array as collection buffer.  We could do similar in GPU,
	// but we defer the actual memory management strategy to the launcher function.
	InteractionResult *gtops = new InteractionResult[par.topmost];
	std::memset(gtops, 0, par.topmost * sizeof(InteractionResult));

	if (par.gold)
		process_gold_1(plkr, gtops, par);
	else
		process_gpu_1(plkr, gtops, par);
	printInteractions(plkr, gtops, par.topmost);

	delete[] gtops;

	delete plkr;

	printf("Cleaned\n");
	return 0;
}

static bool getParams(int argc, char *argv[], struct _paramP1 &opar)
{
	struct _paramP1 par = {
			DEFAULT_GPUNUM,
			DEFAULT_TOPMOST,
			false,
			false,
			false,
			false,
			"",
			DEFAULT_BLOCK_SIDE,
			DEFAULT_BLOCK_SIDE,
			DEFAULT_MAXRESULTS,
			FFUNC_ENTROPY,
			DEFAULT_MISS_SNPS,
			DEFAULT_MISS_SAMP,
			false,
	};
	bool ok = true;

	for (int j = 1; j < argc; ++j)
	{
		string argu(argv[j]);
		if (argu == "-g" || argu == "--gold")
			par.gold = true;
		else if (argu == "-t" || argu == "--alternate-grid")
			par.alt = true;
		else if (argu == "-G" || argu == "--gpu-num")
			par.gpuNum = std::atoi(argv[++j]);
		else if (argu == "-D" || argu == "--print-diagonal")
			par.intermediate = true;
		else if (argu == "-f" || argu == "--fisher-pvals")
			par.ffun = FFUNC_FISHERPV;
		else if (argu == "-i" || argu == "--input-root")
			par.name = argv[++j];
		else if (argu == "-N" || argu == "--topmost")
			par.topmost = atoi(argv[++j]);
		else if (argu == "--block-sizes")
		{
			par.blkSzX = atoi(argv[++j]);
			par.blkSzY = atoi(argv[++j]);
		}
		else if (argu == "--max-grid-area")
			par.maxGridBlocks = atoi(argv[++j]);
		else if (argu == "--shared-memory")
			par.shmem = true;
		else if (argu == "--miss-snps")
			par.qaMissSNPs = atof(argv[++j]);
		else if (argu == "--miss-samples")
			par.qaMissSamp = atof(argv[++j]);
		else
		{
			clog << "Unknown option: " << argu << endl;
			ok = false;
		}
	}
	opar = par;
	if ((par.blkSzX*par.blkSzY) % 32 != 0)
	{
		cerr << "Block size not multiple of war size (32)" << std::endl;
		ok = false;
	}
	if (par.name == "")
	{
		cerr << "Insufficient arguments" << std::endl;
		ok = false;
	}

	if (!ok)
		printHelp();
	return ok;
}

static void printHelp(void)
{
	clog << PROGNAME << " -i <input-root> [options]\n\n"
			"   -i | --input-root      BASE Use BASE as full path and filename for .bed,.bim,.fam\n\n"
			"   -g | --gold                 Process \"gold\" on CPU  [no]\n"
			"   -f | --fisher-pvals         Computes Fisher P-value instead of Entropy [no]\n"
			"   -G | --gpu-num         N    Use GPU number N  [0]\n"
			"   -t | --alternate-grid       For some block sizes, use different layout during init & entropy  [no]\n"
			"   -D | --print-diagonal       Print a few results from time to time  [no]\n"
			"   -N | --topmost         M    Maintain a list of M results  [" << DEFAULT_TOPMOST << "]\n\n"
			"        --block-sizes     N M  Use block size (N,M)  [" << DEFAULT_BLOCK_SIDE << "," << DEFAULT_BLOCK_SIDE << "]\n"
			"        --max-grid-area   N    Use max N x 1024 x 1024 results  [" << DEFAULT_MAXRESULTS << "]\n"
			"        --shared-memory        Use shared memory kernel [no]\n"
			"        --miss-snps       F    Max missing values fraction for SNPs [" << DEFAULT_MISS_SNPS << "]>\n"
			"        --miss-samples    F    Max missing values fraction for samples  [" << DEFAULT_MISS_SAMP << "]\n"
			<< endl;
}

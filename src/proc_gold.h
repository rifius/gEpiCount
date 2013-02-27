/*
 * proc_gold.cpp
 *
 *  Created on: 20/12/2011
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
#ifndef PROC_GOLD_H
#define PROC_GOLD_H

#include <cmath>
#include <limits>
#include <cstring>
#include <cstdlib>
#include <string>

#include "../inc/gEpiCount.h"
#include "misc/Timer.h"
#include "misc/bitops.h"
#include "misc/print_misc.h"

template<typename T>
inline count_type4 binFunc(const T *snpA, const T *snpB, const T *mask0, const T *mask1, int func,
		int nElements) {
	count_type4 r;
	T c;
	r.w = r.x = r.y = r.z = 0;
	for (int j = 0; j < nElements; j++) {
		switch (func) {
		case BFUNC_AND:
			c = (snpA[j] & snpB[j]);
			break;
		case BFUNC_NAND:
			c = (~snpA[j] & snpB[j]);
			break;
		case BFUNC_ANDN:
			c = (snpA[j] & ~snpB[j]);
			break;
		case BFUNC_NANDN:
			c = (~(snpA[j] | snpB[j]));
			break;
		case BFUNC_XOR:
			c = (snpA[j] ^ snpB[j]);
			break;
		default:
			c = 0;
			break;
		}
		r.w += bitsElement(c & mask0[j]);
		r.x += bitsElement(c & mask1[j]);
		r.y += bitsElement(mask0[j]);
		r.z += bitsElement(mask1[j]);
	}
	return r;
}

template<typename T> inline count_type4 binFunc(const T *snpA, const T *snpB, const T *mask0,
		const T *mask1, int func, int nElements, T *out) {
	count_type4 r;
	T c;
	r.w = r.x = r.y = r.z = 0;
	for (int j = 0; j < nElements; j++) {
		switch (func) {
		case BFUNC_AND:
			c = (snpA[j] & snpB[j]);
			break;
		case BFUNC_NAND:
			c = (~snpA[j] & snpB[j]);
			break;
		case BFUNC_ANDN:
			c = (snpA[j] & ~snpB[j]);
			break;
		case BFUNC_NANDN:
			c = (~(snpA[j] | snpB[j]));
			break;
		case BFUNC_XOR:
			c = (snpA[j] ^ snpB[j]);
			break;
		default:
			c = 0;
			break;
		}
		r.w += bitsElement(c & mask0[j]);
		r.x += bitsElement(c & mask1[j]);
		r.y += bitsElement(mask0[j]);
		r.z += bitsElement(mask1[j]);

		out[j] = c & (mask0[j] | mask1[j]);
	}
	return r;
}

#endif	/* PROC_GOLD_H */

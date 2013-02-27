/*
 * bitops.h
 *
 *  Created on: 03/01/2012
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

#ifndef BITOPS_H_
#define BITOPS_H_

#define		NSHORTS		65536
static char 			bits_in_i2[NSHORTS];
static bool 			_preloaded = false;

// Preload bit count array required for bitsElements
static void _do_preload(void)
{
	if (_preloaded)
		return;
	for (int j = 0; j < NSHORTS; j++)
	{
		int count = 0;
		int k = j;
		while (k)
		{
			count++;
			k &= (k - 1);
		}
		bits_in_i2[j] = count;
	}
	_preloaded = true;
}

// Count bits set on an element of type T.  Requires initialization by calling _do_preload()
// This lookup is the fastest version from many.
template <typename T> static inline int bitsElement(T a)
{
	int count = 0;
	for (unsigned int j = 0; j < sizeof(T); j += 2)
	{
		count += (int) bits_in_i2[(unsigned short) (a & (T) 0xffffu)];
		a >>= 16;
	}
	return count;
}



#endif /* BITOPS_H_ */

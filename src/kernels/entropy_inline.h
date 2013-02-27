/*
 * entropy_inline.h
 *
 *  Created on: 16/02/2012
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

#ifndef ENTROPY_INLINE_H_
#define ENTROPY_INLINE_H_

// Entropy delta computation.  I have left out checking for cases with denominator 0 as this signals
// a likely error in the data.  This will return NAN.
inline __host__ __device__ float entropyDelta4(count_type4 rr)
{

//  Attempt to produce faster code by Manuel Ujaldon.
//	float m = rr.y + rr.z;
//	float xx = rr.w + rr.x;
//	float ret = 0.0f;
//	float a2;
//
////	if (m > 0.0f)
////	{
//		if (xx > 0.0f && xx < m)
//		{
//			a2 = xx / m;
//			ret += a2 * log2(a2) + (1.0f - a2) * log2(1.0f - a2);
//		}
//		if (rr.y > 0 && rr.y > rr.w)
//		{
//			a2 = (float) rr.w / (float) rr.y;
//			ret -= ((float) rr.w * log2(a2) + (float) (rr.y - rr.w) * log2(1.0f - a2)) / m;
//		}
//		if (rr.z > 0 && rr.z > rr.x)
//		{
//			a2 = (float) rr.x / (float) rr.z;
//			ret -= ((float) rr.x * log2(a2) + (float) (rr.z - rr.x) * log2(1.0f - a2)) / m;
//		}
////	}

		float m = rr.y + rr.z;
		float xx = rr.w + rr.x;
		float ret = 0.0f;
	//	if (m > 0.0f)
	//	{
#ifdef __CUDA_ARCH__
			if (xx > 0.0f && xx < m)
				ret += (xx/m) * __log2f(xx/m) + ((m - xx)/m) * __log2f((m - xx)/m);
			if (rr.y > 0 && rr.y > rr.w && rr.w > 0)
				ret -= ((float) rr.w/m) * __log2f((float) rr.w/(float) rr.y) + ((float) (rr.y - rr.w)/m) * __log2f((float) (rr.y - rr.w)/(float) rr.y);
			if (rr.z > 0 && rr.z > rr.x && rr.x > 0)
				ret -= ((float) rr.x/m) * __log2f((float) rr.x/(float) rr.z) + ((float) (rr.z - rr.x)/m) * __log2f((float) (rr.z - rr.x)/(float) rr.z);
#else
			if (xx > 0.0f && xx < m)
				ret += (xx/m) * log2(xx/m) + ((m - xx)/m) * log2((m - xx)/m);
			if (rr.y > 0 && rr.y > rr.w && rr.w > 0)
				ret -= ((float) rr.w/m) * log2((float) rr.w/(float) rr.y) + ((float) (rr.y - rr.w)/m) * log2((float) (rr.y - rr.w)/(float) rr.y);
			if (rr.z > 0 && rr.z > rr.x && rr.x > 0)
				ret -= ((float) rr.x/m) * log2((float) rr.x/(float) rr.z) + ((float) (rr.z - rr.x)/m) * log2((float) (rr.z - rr.x)/(float) rr.z);
#endif
	//	}
#endif
	return ret;
}

#endif /* ENTROPY_INLINE_H_ */

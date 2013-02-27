/*
 * PlinkReader.cpp
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
#ifndef SAMPLECLASSINFO_H
#define SAMPLECLASSINFO_H

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <stdexcept>

template <class T> class PlinkReader;

template <class ET> class SampleClassInfo
{
private:
	int cid;
	int numSamples;
	int numSamplesClass;
	ET *smask;
public:
	friend class PlinkReader<ET>;
	SampleClassInfo(int aclassId, int nSamples)
	{
		this->numSamples = nSamples;
		this->numSamplesClass = 1;
		this->cid = aclassId;
		int ne = ((nSamples + PlinkReader<ET>::SAMPLES_ELEMENT - 1) / PlinkReader<ET>::SAMPLES_ELEMENT);
		this->smask = new ET[ne];
		memset(this->smask, 0, ne * sizeof(ET));
	}

	virtual ~SampleClassInfo()
	{
		delete[] this->smask;
	}

    int classId() const
    {
        return cid;
    }

    ET *mask() const
    {
        return smask;
    }

    int mumSamplesClass() const
    {
        return numSamplesClass;
    }
    void incrementCounter()
    {
    	numSamplesClass++;
    }
};

#endif /* SAMPLECLASSINFO_H */

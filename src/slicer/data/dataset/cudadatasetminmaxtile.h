#ifndef CUDADatasetMinMaxTile_H
#define CUDADatasetMinMaxTile_H

#include <iostream>
#include <cuda_runtime_api.h>
#include <boost/iostreams/device/mapped_file.hpp>
#include <QVector2D>

class Seismic3DDataset;

class CUDADatasetMinMaxTile {
public:
	CUDADatasetMinMaxTile(int w, int h, int tileSize);
	~CUDADatasetMinMaxTile();

	QVector2D minMax(int d0, int d1,Seismic3DDataset *seismic, int channel);
private:
	void allocate(int typeByteSize);
	void free();

	size_t round(size_t numToRound, size_t multiple);
private:
	int m_w;
	int m_h;
	int m_tileSize;

	int m_d0;
	int m_d1;
	int m_depth;
	int m_dimV;

	void *seismicDeviceBuffer;
	bool m_isAllocated;
	int m_allocateByteSize;
};

#endif

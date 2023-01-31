#ifndef CUDARGTTile_H
#define CUDARGTTile_H

#include <iostream>
#include <cuda_runtime_api.h>
#include <boost/iostreams/device/mapped_file.hpp>

class CUDAImagePaletteHolder;
class Seismic3DDataset;
class DatasetBlocTile;

class CUDARGTTile {
public:
	CUDARGTTile(int w, int h, int tileSize);
	~CUDARGTTile();

	void fillHostBuffer(Seismic3DDataset *seismic, int channelS,
			Seismic3DDataset *rgt, int channelT, int d0, int d1);
	void run(short z, uint window,short * isoVal,short *attrVal);
	void writeResult();
private:
	void allocate();
	void free();

	size_t round(size_t numToRound, size_t multiple);
	size_t blocSize();
private:
	int m_w;
	int m_h;
	int m_tileSize;

	int m_d0;
	int m_d1;
	int m_depth;

	void *seismicDeviceBuffer;
	void *rgtDeviceBuffer;



	cudaStream_t stream;

	DatasetBlocTile *seismicTile;
	bool registerSeismicTile;
	DatasetBlocTile *rgtTile;
	bool registerRGTTile;

	void *seismicData;
	void *rgtData;
};

#endif

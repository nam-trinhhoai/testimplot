#ifndef CUDARGBTile_H
#define CUDARGBTile_H

#include <iostream>
#include <cuda_runtime.h>
#include <boost/iostreams/device/mapped_file.hpp>
#include <cufft.h>

class CUDAImagePaletteHolder;
class Seismic3DDataset;
class DatasetBlocTile;

class CUDARGBTile {
public:
	CUDARGBTile(int w, int h, int tileSize,int window);
	~CUDARGBTile();

	void fillHostBuffer(Seismic3DDataset *seismic, int channelS, Seismic3DDataset *rgt, int channelT, int d0, int d1);
	void run(short z,unsigned int f1, unsigned int f2,unsigned int f3,short *isoVal, float *f1Val, float *f2Val,
			float *f3Val);
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

	int m_window;

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

	cufftComplex *deviceOutputData;
	cufftHandle handle;
	float *data_f;
};

#endif

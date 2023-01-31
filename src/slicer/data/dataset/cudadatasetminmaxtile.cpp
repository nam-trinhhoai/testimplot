#include  "cudadatasetminmaxtile.h"
#include "seismic3ddataset.h"
#include <cuda_runtime.h>
#include "cuda_optvolume.h"
#include <iostream>
#include "datasetbloctile.h"

#include "cuda_common_helpers.h"

CUDADatasetMinMaxTile::CUDADatasetMinMaxTile(int w, int h, int tileSize) {
	m_w = w;
	m_h = h;
	m_tileSize = tileSize;
	seismicDeviceBuffer = nullptr;

	m_isAllocated = false;
	m_allocateByteSize = 0;
	//allocate();
}

CUDADatasetMinMaxTile::~CUDADatasetMinMaxTile() {
	free();
}

void CUDADatasetMinMaxTile::allocate(int typeByteType) {
	if (!m_isAllocated || typeByteType>m_allocateByteSize) {
		free();
		size_t subBlocSize = m_tileSize * m_w * m_h * typeByteType;
		checkCudaErrors(cudaMalloc(&seismicDeviceBuffer, subBlocSize));
		m_allocateByteSize = typeByteType;
		m_isAllocated = true;
	}
}

void CUDADatasetMinMaxTile::free() {
	if (m_isAllocated) {
		cudaFree(seismicDeviceBuffer);
	}
}

template<typename InputType>
struct MinMaxKernel {
	static void run(void* buffer, std::size_t blocSize, float& min, float& max) {
		computeMinMaxOptimizedOpt<InputType>((InputType*) buffer,  blocSize, min, max);
	}
};

//state 2: wait state 1
QVector2D CUDADatasetMinMaxTile::minMax(int d0, int d1,
		Seismic3DDataset *seismic, int channel) {
	allocate(seismic->sampleType().byte_size());

	m_d0 = d0;
	m_d1 = d1;

	m_depth = m_d1 - m_d0;

	float min, max; // potential issue with double cubes
	DatasetBlocTile tile(seismic->path(), channel, seismic->headerLength(), m_w, m_h,
			m_d0, m_d1, seismic->dimV(), seismic->sampleType());
	size_t blocSize=m_w * m_h * m_depth;

//	auto cuda_retval = cudaHostRegister(tile.buffer(),tile.memoryCost(),cudaHostRegisterMapped);
//	if (cuda_retval != cudaSuccess)
//		std::cout<< cudaGetErrorString (cuda_retval) + std::string (" (cudaHostRegister)")<<std::endl;

	cudaMemcpy(seismicDeviceBuffer, (char *)tile.buffer()+tile.dataStartOffset() ,blocSize*seismic->sampleType().byte_size(),
			cudaMemcpyHostToDevice);
	SampleTypeBinder binder(seismic->sampleType());
	binder.bind<MinMaxKernel>(seismicDeviceBuffer, blocSize, min, max);

//	cuda_retval=cudaHostUnregister(tile.buffer());
//	if (cuda_retval != cudaSuccess)
//			std::cout<< cudaGetErrorString (cuda_retval) + std::string (" (cudaHostRegister)")<<std::endl;

	return QVector2D(min,max);
}

size_t CUDADatasetMinMaxTile::round(size_t numToRound, size_t multiple) {
	if (multiple == 0)
		return numToRound;

	int remainder = numToRound % multiple;
	if (remainder == 0)
		return numToRound;

	return numToRound - remainder;
}


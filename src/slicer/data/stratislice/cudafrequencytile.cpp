#include  "cudafrequencytile.h"
#include "cudaimagepaletteholder.h"
#include "seismic3ddataset.h"
#include <cuda_runtime.h>
#include "cuda_volume.h"
#include "cuda_optvolume.h"
#include "cuda_common_helpers.h"
#include <iostream>
#include <QElapsedTimer>

#include "datasetbloctile.h"
#include "datasetbloccache.h"


CUDAFrequencyTile::CUDAFrequencyTile(int w, int h, int tileSize, int window) {
	m_w = w;
	m_h = h;
	m_tileSize = tileSize;
	m_window = window;

	seismicDeviceBuffer = rgtDeviceBuffer = nullptr;
	allocate();

	registerSeismicTile=false;
	registerRGTTile=false;
}

CUDAFrequencyTile::~CUDAFrequencyTile() {
	free();
}

//This container is a state machine (eg: not a class)
size_t CUDAFrequencyTile::blocSize() {
	return m_tileSize * m_w * m_h * sizeof(short);
}

void CUDAFrequencyTile::allocate() {
	size_t subBlocSize = blocSize();

	checkCudaErrors(cudaMalloc(&seismicDeviceBuffer, subBlocSize));
	checkCudaErrors(cudaMalloc(&rgtDeviceBuffer, subBlocSize));

	size_t planeSize = m_tileSize * m_w;
	cudaMalloc(&data_f, m_w * m_window * m_tileSize * sizeof(float));
	cudaMalloc(&deviceOutputData,
			m_w * m_tileSize * (m_window / 2 + 1) * sizeof(cufftComplex));

	cudaStreamCreate(&stream);
	// --- Batched 1D FFTs
	int rank = 1;                           // --- 1D FFTs
	int n[] = { (int) m_window };           // --- Size of the Fourier transform
	int istride = 1, ostride = 1; // --- Distance between two successive input/output elements
	int idist = m_window, odist = (m_window / 2 + 1); // --- Distance between batches
	int inembed[] = { 0 }; // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 }; // --- Output size with pitch (ignored for 1D transforms)
	// --- Number of batched executions
	cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride,
			odist, CUFFT_R2C, m_w * m_tileSize);
	cufftSetStream(handle, stream);
}

//state 3; wait state 2
void CUDAFrequencyTile::free() {
	cudaFree(seismicDeviceBuffer);
	cudaFree(rgtDeviceBuffer);

	cudaFree(data_f);
	cudaFree(deviceOutputData);

	cufftDestroy(handle);

	cudaStreamDestroy(stream);
}
//state 2: wait state 1
void CUDAFrequencyTile::writeResult() {
	cudaStreamSynchronize(stream);

//	cudaHostUnregister(seismicData);
//	cudaHostUnregister(rgtData);

//	if(registerRGTTile)
//	{
//		DatasetBlocCache::getInstance()->insert(rgtTile->key(),rgtTile);
//	}
//	if(registerSeismicTile)
//	{
//		DatasetBlocCache::getInstance()->insert(seismicTile->key(),seismicTile);
//	}
}

//state 1
void CUDAFrequencyTile::run(short z,short *isoVal, float **fVal) {
	hanningAndIsoBlocExtract(stream, (short*) seismicDeviceBuffer,
			(short*) rgtDeviceBuffer, isoVal + m_d0 * m_w, data_f, m_w,
			m_h, m_depth, z, m_window);

	cufftExecR2C(handle, data_f, deviceOutputData);
	optimizedFFTModuleAll(stream, deviceOutputData, fVal , m_d0 * m_w, m_w, m_h, m_depth, m_window);
}

void CUDAFrequencyTile::fillHostBuffer(Seismic3DDataset *seismic, int channelS,
		Seismic3DDataset *rgt, int channelT, int d0, int d1) {
	m_d0 = d0;
	m_d1 = d1;

	m_depth = m_d1 - m_d0;

	DatasetHashKey kSeismic(seismic->path(),m_d0,m_d1);
	seismicTile=DatasetBlocCache::getInstance()->object(kSeismic);
	if(seismicTile==nullptr)
	{
		seismicTile=new DatasetBlocTile(seismic->path(), channelS, seismic->headerLength(), m_w, m_h,
					m_d0, m_d1, seismic->dimV(), seismic->sampleType());
		registerSeismicTile=true;
	}else
		registerSeismicTile=false;

	DatasetHashKey kRgt(rgt->path(),m_d0,m_d1);
	rgtTile=DatasetBlocCache::getInstance()->object(kRgt);
	if(rgtTile==nullptr)
	{
		rgtTile=new DatasetBlocTile(rgt->path(), channelT, rgt->headerLength(), m_w, m_h,
					m_d0, m_d1, rgt->dimV(), rgt->sampleType());
		registerRGTTile=true;
	}else
		registerRGTTile=false;

	rgtData = rgtTile->buffer();

	//We register the buffer for CUDA (will increase transfert rate)
	seismicData = seismicTile->buffer();
	rgtData = rgtTile->buffer();

//	auto cuda_retval=cudaHostRegister(seismicData,seismicTile->memoryCost(),cudaHostRegisterMapped);
//	if (cuda_retval != cudaSuccess)
//			std::cout<< cudaGetErrorString (cuda_retval) + std::string (" (cudaHostRegister)")<<std::endl;
//	cudaHostRegister(rgtData,rgtTile->memoryCost(),cudaHostRegisterMapped);

	size_t blocSize = m_w * m_h * m_depth * sizeof(short);

	cudaMemcpyAsync(seismicDeviceBuffer, (char *)seismicData+seismicTile->dataStartOffset(), blocSize,
			cudaMemcpyHostToDevice, stream);

	cudaMemcpyAsync(rgtDeviceBuffer, (char *)rgtData+rgtTile->dataStartOffset(), blocSize,
			cudaMemcpyHostToDevice, stream);
}


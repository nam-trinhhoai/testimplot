#include  "cudargttile.h"
#include "cudaimagepaletteholder.h"
#include "seismic3ddataset.h"
#include <cuda_runtime.h>
#include "cuda_optvolume.h"
#include "cuda_common_helpers.h"
#include <iostream>
#include "datasetbloctile.h"
#include "datasetbloccache.h"
#include "cuda_common_helpers.h"

CUDARGTTile::CUDARGTTile(int w, int h, int tileSize) {
	m_w = w;
	m_h = h;
	m_tileSize = tileSize;
	seismicDeviceBuffer = rgtDeviceBuffer=nullptr;
	allocate();

	registerSeismicTile=false;
	registerRGTTile=false;
}

CUDARGTTile::~CUDARGTTile() {
	free();
}

//This container is a state machine (eg: not a class)
size_t CUDARGTTile::blocSize() {
	return m_tileSize * m_w * m_h * sizeof(short);
}


void CUDARGTTile::allocate() {
	size_t subBlocSize = blocSize();

	checkCudaErrors(cudaMalloc(&seismicDeviceBuffer,subBlocSize));
	checkCudaErrors(cudaMalloc(&rgtDeviceBuffer, subBlocSize));

	cudaStreamCreate(&stream);
}

//state 3; wait state 2
void CUDARGTTile::free() {
	cudaFree(seismicDeviceBuffer);
	cudaFree(rgtDeviceBuffer);

	cudaStreamDestroy(stream);
}
//state 2: wait state 1
void CUDARGTTile::writeResult() {
	cudaStreamSynchronize(stream);

	//close streams to be reopenned further
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

template<typename SeismicType>
struct CUDARGTTileLevel1Kernel {
	template<typename RgtType>
	struct CUDARGTTileLevel2Kernel {
		static void run(cudaStream_t stream, void* seismicBuffer, void* rgtBuffer, short *attributeArray, short *isovalueArray,
				uint w, uint h, uint d, short val, uint window) {
			attributeAndIsoValueBlocExtractOpt(stream, (SeismicType*)seismicBuffer, (RgtType*)rgtBuffer, attributeArray, isovalueArray, w, h, d, val, window);
		}
	};

	static void run(ImageFormats::QSampleType rgtType, cudaStream_t stream, void* seismicBuffer, void* rgtBuffer,
			short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val, uint window) {
		SampleTypeBinder binder(rgtType);
		binder.bind<CUDARGTTileLevel2Kernel>(stream, seismicBuffer, rgtBuffer, attributeArray, isovalueArray, w, h, d, val, window);
	}
};

//state 1
void CUDARGTTile::run(short z, uint window,short *isoVal, short *attrVal) {
	SampleTypeBinder binder(seismicTile->sampleType());
	binder.bind<CUDARGTTileLevel1Kernel>(rgtTile->sampleType(), stream,seismicDeviceBuffer,
			rgtDeviceBuffer, attrVal + m_d0 * m_w,
			isoVal + m_d0 * m_w, m_w, m_h, m_depth, z, window);
}

size_t CUDARGTTile::round(size_t numToRound, size_t multiple) {
	if (multiple == 0)
		return numToRound;

	int remainder = numToRound % multiple;
	if (remainder == 0)
		return numToRound;

	return numToRound - remainder;
}

void CUDARGTTile::fillHostBuffer(Seismic3DDataset *seismic, int channelS,
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
	seismicData = seismicTile->buffer();

//	auto cuda_retval =cudaHostRegister(seismicData,seismicTile->memoryCost(),cudaHostRegisterMapped);
//	if (cuda_retval != cudaSuccess)
//		std::cout<< cudaGetErrorString (cuda_retval) + std::string (" (cudaHostRegister)")<<std::endl;
//	cudaHostRegister(rgtData,rgtTile->memoryCost(),cudaHostRegisterMapped);

	//Grab the pointer
//	cuda_retval =cudaHostGetDevicePointer(&seismicDeviceBuffer,seismicData,0);
//	if (cuda_retval != cudaSuccess)
//		std::cout<< cudaGetErrorString (cuda_retval) + std::string (" (cudaHostGetDevicePointer)")<<std::endl;
//	cudaHostGetDevicePointer(&rgtDeviceBuffer,rgtData,0);

//	seismicDeviceBuffer=(char *)seismicDeviceBuffer +seismicTile->dataStartOffset();
//	rgtDeviceBuffer=(char *)rgtDeviceBuffer +rgtTile->dataStartOffset();

	size_t blocSizeSeismic = m_w * m_h * m_depth * seismic->sampleType().byte_size();
	size_t blocSizeRgt = m_w * m_h * m_depth * rgt->sampleType().byte_size();
	cudaMemcpyAsync(seismicDeviceBuffer, (char *)seismicData+seismicTile->dataStartOffset(), blocSizeSeismic,
			cudaMemcpyHostToDevice, stream);

	cudaMemcpyAsync(rgtDeviceBuffer, (char *)rgtData+rgtTile->dataStartOffset(), blocSizeRgt,
			cudaMemcpyHostToDevice, stream);
}


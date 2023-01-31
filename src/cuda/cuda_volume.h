#ifndef CUDA_VOLUME_H_
#define CUDA_VOLUME_H_

#include <cuda_runtime_api.h>
#include <cufft.h>

extern "C" short* initCuda3DVolume(short *h_volume,
		const cudaExtent &volumeSize);

extern "C" void blocFFT(short *seismic, short *rgt, short *isovalueArray,
		float *f1Array, float *f2Array, float *f3Array, unsigned int w,
		unsigned int h, unsigned int d, unsigned int val, unsigned int window,
		unsigned int f1, unsigned int f2, unsigned int f3);

extern "C" void blocFFTAll(short *seismic, short *rgt, short *isovalueArray,
		float **fArray, unsigned int w, unsigned int h, unsigned int d,
		unsigned int val, unsigned int window);

extern "C" void optimizedFFTModule(const cudaStream_t &stream,
		cufftComplex *deviceOutputData, float *f1Array, float *f2Array,
		float *f3Array, unsigned int w, unsigned int h, unsigned int d,
		unsigned int window, unsigned int f1, unsigned int f2, unsigned int f3);

extern "C" void optimizedFFTModuleAll(const cudaStream_t &stream,
		cufftComplex *deviceOutputData, float **fArray, int offset,unsigned int w,
		unsigned int h, unsigned int d, unsigned int window);

extern "C" void attributeAndIsoValueBlocExtract(short *volume, short *rgt,
		short *attributeArray, short *isovalueArray, uint w, uint h, uint d,
		short val, uint window);

template<typename InputType>
void isoLineExtract(InputType *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *line);

template<typename InputType>
void byteSwapAndTransposeImageData(InputType *output, InputType *data,
		uint w, uint h, uint c=1);

// test swaps
//void testSwap();

#endif

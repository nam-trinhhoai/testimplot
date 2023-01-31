#ifndef CUDA_OPTVOLUME_H_
#define CUDA_OPTVOLUME_H_

#include <cuda_runtime_api.h>

template<typename SeismicType, typename RgtType>
void attributeAndIsoValueBlocExtractOpt(const cudaStream_t &stream,
		SeismicType *volume, RgtType *rgt, short *attributeArray, short *isovalueArray,
		uint w, uint h, uint d, short val, uint window);

template<typename InputType>
void computeMinMaxOptimizedOpt(InputType *cuda_image_array, size_t N,
		float &min, float &max);

extern "C" void hanningAndIsoBlocExtract(const cudaStream_t &stream,short *seismic, short *rgt,
		short *isovalueArray, float *data_f, uint w, uint h, uint d, short val,
		uint window);

#endif

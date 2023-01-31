#ifndef CUDA_ALGO_H_
#define CUDA_ALGO_H_

#include <cuda_runtime_api.h>

template<typename T>  void computeImageHistogram(T *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio);

//Min/max
template<typename T>  void computeMinMaxOptimized(T *cuda_image_array,size_t N,float & min,float & max);

template<typename T>  void computeMinMaxOptimizedMultiChannels(T *cuda_image_array,size_t N, float* ranges, size_t channelCount);

template<typename T> void renderInline(T *cuda_image_array,T * volumeArray, const cudaExtent &extent,uint w, uint h, uint pos);
template<typename T> void renderXline(T *cuda_image_array, T * volumeArray, const cudaExtent &extent, uint d, uint h,uint pos);


#endif

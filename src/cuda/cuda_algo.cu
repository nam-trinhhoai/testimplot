#include "cuda_algo.h"
#include "cuda_algo_kernel.cuh"

#include "cuda_common_helpers.h"
#include <limits>
#include <vector>
#include <float.h>
#include <iostream>

//https://stackoverflow.com/questions/24475872/cuda-reduction-to-find-the-maximum-of-an-array
//https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<typename T>
void computeMinMaxOptimized(T *cuda_image_array,size_t N, float & min,float & max)
{
	const unsigned int blockSize=512;
	int NumThreads  = (N < blockSize) ? nextPow2(N) : blockSize;
	int NumBlocks   = (N + NumThreads - 1) / NumThreads;

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (NumThreads <= 32) ? 2 * NumThreads * sizeof(int) : NumThreads * sizeof(int);
	smemSize=2*smemSize;

	// reduce2 type kernel
	void * tempVector;
	cudaMalloc(&tempVector,2*NumBlocks * sizeof(T));

	dim3 dimBlock(NumThreads,1);
	dim3 dimGrid(NumBlocks,1);
	reduce_minmax_kernel<<<dimGrid, dimBlock, smemSize>>>(cuda_image_array,(T *)tempVector, N,NumBlocks,std::numeric_limits<T>::lowest(),std::numeric_limits<T>::max());
	checkCudaErrors(cudaDeviceSynchronize());
	T *res=new T[2*NumBlocks];
	cudaMemcpy((void *) res,tempVector,2*NumBlocks * sizeof(T),cudaMemcpyDeviceToHost);

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	max = std::numeric_limits<T>::lowest();
	min = std::numeric_limits<T>::max();
	for (int i=0; i<NumBlocks; i++)
	{
		if(max<res[i])
			max =res[i];

		if(min>res[i+NumBlocks])
			min =res[i+NumBlocks];
	}
	delete[] res;
	cudaFree(tempVector);
}

template void computeMinMaxOptimized(signed char *,size_t ,  float & ,float & );
template void computeMinMaxOptimized(unsigned char *,size_t ,  float & ,float & );
template void computeMinMaxOptimized(short *,size_t ,  float & ,float & );
template void computeMinMaxOptimized(unsigned short *,size_t ,  float & ,float & );
template void computeMinMaxOptimized(int *,size_t ,  float & ,float & );
template void computeMinMaxOptimized(unsigned int *,size_t ,  float & ,float & );
template void computeMinMaxOptimized(float *,size_t ,  float & ,float & );
template void computeMinMaxOptimized(double *,size_t ,  float & ,float & );

template<typename T>
void computeMinMaxOptimizedMultiChannels(T *cuda_image_array,size_t N, float* ranges, size_t channelCount)
{
	const unsigned int blockSize=512;
	int NumThreads  = (N < blockSize) ? nextPow2(N) : blockSize;
	int NumBlocks   = (N + NumThreads - 1) / NumThreads;

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (NumThreads <= 32) ? 2 * NumThreads * sizeof(int) : NumThreads * sizeof(int);
	smemSize=2*channelCount*smemSize;

	// reduce2 type kernel
	void * tempVector;
	cudaMalloc(&tempVector,2*channelCount*NumBlocks * sizeof(T));

	dim3 dimBlock(NumThreads,1);
	dim3 dimGrid(NumBlocks,1);
	reduce_minmax_kernel_multi_channels<<<dimGrid, dimBlock, smemSize>>>(cuda_image_array,(T *)tempVector, N,NumBlocks,std::numeric_limits<T>::lowest(),std::numeric_limits<T>::max(), channelCount);
	checkCudaErrors(cudaDeviceSynchronize());
	T *res=new T[2*channelCount*NumBlocks];
	cudaMemcpy((void *) res,tempVector,2*channelCount*NumBlocks * sizeof(T),cudaMemcpyDeviceToHost);

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	float max, min;
	for (size_t k=0; k<channelCount; k++) {
		max = std::numeric_limits<T>::lowest();
		min = std::numeric_limits<T>::max();

		for (int i=0; i<NumBlocks; i++)
		{
			if(max<res[i+NumBlocks*k*2])
				max =res[i+NumBlocks*k*2];

			if(min>res[i+NumBlocks*(k*2+1)])
				min =res[i+NumBlocks*(k*2+1)];
		}
		ranges[2*k+1] = max;
		ranges[2*k] = min;
	}
	delete[] res;
	cudaFree(tempVector);
}

template void computeMinMaxOptimizedMultiChannels(signed char *,size_t , float*, size_t );
template void computeMinMaxOptimizedMultiChannels(unsigned char *,size_t , float*, size_t );
template void computeMinMaxOptimizedMultiChannels(short *,size_t , float*, size_t );
template void computeMinMaxOptimizedMultiChannels(unsigned short *,size_t , float*, size_t );
template void computeMinMaxOptimizedMultiChannels(int *,size_t , float*, size_t );
template void computeMinMaxOptimizedMultiChannels(unsigned int *,size_t , float*, size_t );
template void computeMinMaxOptimizedMultiChannels(float *,size_t , float*, size_t );
template void computeMinMaxOptimizedMultiChannels(double *,size_t , float*, size_t );

template<typename T>
void computeImageHistogram(T *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio)
{
	dim3 block(32, 4);
	dim3 grid(16, 16);
	int total_blocks = grid.x * grid.y;

	// allocate partial histogram
	unsigned int *d_part_hist;
	cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

	histogram_smem_atomics<<<grid, block>>>(cuda_image_array,
	        w,
	        h,
	        d_part_hist,min,ratio);

	dim3 block2(128);
	dim3 grid2((NUM_BINS + block.x - 1) / block.x);

	histogram_final_accum<<<grid2, block2>>>(
		d_part_hist,
		total_blocks,
		hist);

	cudaFree(d_part_hist);
}

template void computeImageHistogram(signed char *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio);
template void computeImageHistogram(unsigned char *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio);
template void computeImageHistogram(short *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio);
template void computeImageHistogram(unsigned short *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio);
template void computeImageHistogram(int *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio);
template void computeImageHistogram(unsigned int *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio);
template void computeImageHistogram(float *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio);
template void computeImageHistogram(double *cuda_image_array,unsigned int *hist,uint w, uint h,float min, float ratio);


template<typename T>
 void renderInline(T * cuda_image_array, T * volumeArray, const cudaExtent &extent,uint w, uint h,
		uint pos) {
	const int bloc_size = 32; // 256 threads per block
	dim3 dimBlock(bloc_size, bloc_size);
	dim3 dimGrid((w - 1) / bloc_size + 1, (h - 1) / bloc_size + 1);
	renderInline_kernel<<<dimGrid, dimBlock>>>(cuda_image_array,volumeArray, w, h, pos,extent.width,extent.height);
}
template<typename T>
void renderXline(T *cuda_image_array,T * volumeArray, const cudaExtent &extent, uint h, uint d,
		uint pos) {
	const int bloc_size = 32; // 256 threads per block
	dim3 dimBlock(bloc_size, bloc_size);
	dim3 dimGrid((h - 1) / bloc_size + 1, (d - 1) / bloc_size + 1);
	renderXline_kernel<<<dimGrid, dimBlock>>>(cuda_image_array, volumeArray,h, d,pos,extent.width,extent.height);
}
template void renderInline(short * cuda_image_array, short * volumeArray, const cudaExtent &extent,uint w, uint h,
		uint pos);
template void renderInline(float * cuda_image_array, float * volumeArray, const cudaExtent &extent,uint w, uint h,
		uint pos);

template void renderXline(short *cuda_image_array,short * volumeArray, const cudaExtent &extent, uint h, uint d,
		uint pos);
template void renderXline(float *cuda_image_array,float * volumeArray, const cudaExtent &extent, uint h, uint d,
		uint pos);


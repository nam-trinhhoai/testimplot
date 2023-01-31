#ifndef CUDA_OPTVOLUME_KERNEL_CUH_
#define CUDA_OPTVOLUME_KERNEL_CUH_

#include <cuda.h>
#include <cufft.h>

#include "cuda_swap_kernel.cuh"

__device__ short swap2Bytes(short val) {
	return (((val) >> 8) & 0x00FF) | (((val) << 8) & 0xFF00);
}

template<typename InputType>
__device__ void reduce_minmax_kernel_opt_device(InputType * input,InputType *d_out, InputType* minMapTemp, size_t size,
		int numBlocks, InputType absMin, InputType absMax) {
	int tid = threadIdx.x;                              // Local thread index
	int localX = blockIdx.x * blockDim.x + threadIdx.x;// Global thread index

	// --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
	minMapTemp[tid] = (localX<size) ? byteswap(input[localX]) : absMin;
	minMapTemp[tid+blockDim.x] = (localX<size) ? byteswap(input[localX]) :absMax;

	// --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

	// --- Reduction in shared memory. Only half of the threads contribute to reduction.
	for (unsigned int s=blockDim.x/2; s>0; s>>=1)
	{
		if (tid < s) {
			if(minMapTemp[tid]<minMapTemp[tid + s])
			minMapTemp[tid] =minMapTemp[tid + s];

			int offset=tid+blockDim.x;
			if(minMapTemp[offset]>minMapTemp[offset + s])
			minMapTemp[offset] =minMapTemp[offset + s];
		}
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
		__syncthreads();
	}

	if (tid == 0) {
		d_out[blockIdx.x] = minMapTemp[0];
		d_out[numBlocks + blockIdx.x] = minMapTemp[blockDim.x];
	}
}

template<typename InputType>
__global__ void reduce_minmax_kernel_opt(InputType * input,InputType *d_out,size_t size,int numBlocks, InputType absMin, InputType absMax);

template<>
__global__ void reduce_minmax_kernel_opt(signed char * input,signed char *d_out,size_t size,int numBlocks, signed char absMin, signed char absMax) {
	extern __shared__ signed char minMapTempSC[];
	reduce_minmax_kernel_opt_device(input, d_out, minMapTempSC, size, numBlocks, absMin, absMax);
}

template<>
__global__ void reduce_minmax_kernel_opt(unsigned char * input,unsigned char *d_out,size_t size,int numBlocks, unsigned char absMin,unsigned char absMax) {
	extern __shared__ unsigned char minMapTempUC[];
	reduce_minmax_kernel_opt_device(input, d_out, minMapTempUC, size, numBlocks, absMin, absMax);
}

template<>
__global__ void reduce_minmax_kernel_opt(short * input,short *d_out,size_t size,int numBlocks,short absMin,short absMax) {
	extern __shared__ short minMapTempSS[];
	reduce_minmax_kernel_opt_device(input, d_out, minMapTempSS, size, numBlocks, absMin, absMax);
}

template<>
__global__ void reduce_minmax_kernel_opt(unsigned short * input,unsigned short *d_out,size_t size,int numBlocks, unsigned short absMin, unsigned short absMax) {
	extern __shared__ unsigned short minMapTempUS[];
	reduce_minmax_kernel_opt_device(input, d_out, minMapTempUS, size, numBlocks, absMin, absMax);
}

template<>
__global__ void reduce_minmax_kernel_opt(int * input, int *d_out,size_t size,int numBlocks,int absMin,int absMax) {
	extern __shared__ int minMapTempSI[];
	reduce_minmax_kernel_opt_device(input, d_out, minMapTempSI, size, numBlocks, absMin, absMax);
}

template<>
__global__ void reduce_minmax_kernel_opt(unsigned int * input,unsigned int *d_out,size_t size,int numBlocks,unsigned int absMin,unsigned int absMax) {
	extern __shared__ unsigned int minMapTempUI[];
	reduce_minmax_kernel_opt_device(input, d_out, minMapTempUI, size, numBlocks, absMin, absMax);
}

template<>
__global__ void reduce_minmax_kernel_opt(float * input,float *d_out,size_t size,int numBlocks,float absMin,float absMax) {
	extern __shared__ float minMapTempF[];
	reduce_minmax_kernel_opt_device(input, d_out, minMapTempF, size, numBlocks, absMin, absMax);
}

template<>
__global__ void reduce_minmax_kernel_opt(double * input,double *d_out,size_t size,int numBlocks,double absMin,double absMax) {
	extern __shared__ double minMapTempD[];
	reduce_minmax_kernel_opt_device(input, d_out, minMapTempD, size, numBlocks, absMin, absMax);
}

//Iso value extraction
template<typename SeismicType, typename RgtType>
__global__ void attrAndIsoSurfaceExtractOpt_kernel(RgtType *rgt, short *isoValueSurface,
		SeismicType *seismic, short *output, uint w, uint h, uint d, unsigned int val,
		uint window) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.y * blockDim.y + threadIdx.y;
	RgtType val1, val2;
	if (x < w && z < d) {
		short y = 0, y1 = 1;
		//isovalue extraction
		val1 = byteswap(rgt[z * w * h + x * h + y]);
		val2 = byteswap(rgt[z * w * h + x * h + y1]);
		while (!(val1 <= val && val2 >= val) && y < h - 1) {
			y++;
			y1++;
			val1 = val2;
			val2 = byteswap(rgt[z * w * h + x * h + y1]);
		}
		isoValueSurface[z * w + x] = y;

		//Mean
		const unsigned short window2 = (window - 1) / 2;
		int count = 0;
		float sum = 0;
		for (int iy = -window2; iy <= window2; iy++) {
			int pos = y + iy;
			if (pos >= 0 && pos < h) {
				sum += byteswap(seismic[z * w * h + x * h + pos]);
				count++;
			}
		}
		if (count != 0)
			output[z * w + x] = (short) (sum / count);
		else
			output[z * w + x] = 0; //define no data value
	}
}

//Iso value extraction
__global__ void hanningAndisoSurfaceExtractOpt_kernel(short *rgt, short *isoValueSurface,
		short *seismic, float *output, uint w, uint h, uint d, unsigned int val,
		uint window) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.y * blockDim.y + threadIdx.y;
	short val1, val2;
	if (x < w && z < d) {
		short y = 0, y1 = 1;
		//isovalue extraction
		val1 = swap2Bytes(rgt[z * w * h + x * h + y]);
		val2 = swap2Bytes(rgt[z * w * h + x * h + y1]);
		while (!(val1 <= val && val2 >= val) && y < h - 1) {
			y++;
			y1++;
			val1 = val2;
			val2 = swap2Bytes(rgt[z * w * h + x * h + y1]);
		}
		isoValueSurface[z * w + x] = y;

		const unsigned short window2 = (window - 1) / 2;
		int index = 0;
		for (int iy = -window2; iy <= window2; iy++) {
			float wt = 0.5f* (1.0f+ cos(6.28318f* ((float) index/ (float) (window - 1)- 0.5f)));
			int pos = y + iy;
			if (pos >= 0 && pos < h)
				output[z * w * window + x * window + index] = wt
						* swap2Bytes(seismic[z * w * h + x * h + pos]);
			else
				output[z * w * window + x * window + index] =0;
			index++;
		}
	}
}

#endif

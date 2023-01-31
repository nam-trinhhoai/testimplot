#ifndef CUDA_ALGO_KERNEL_CUH_
#define CUDA_ALGO_KERNEL_CUH_

#include <limits>

template<typename T>
__global__ void reduce_minmax_kernel(T * input,T *d_out,size_t size,int numBlocks,T absMin,T absMax) {
	int tid = threadIdx.x;                              // Local thread index
	int localX = blockIdx.x * blockDim.x + threadIdx.x;// Global thread index

	extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
	T *minMapTemp = reinterpret_cast<T *>(my_smem);

	// --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
	minMapTemp[tid] = (localX<size) ? input[localX] : absMin;
	minMapTemp[tid+blockDim.x] = (localX<size) ? input[localX] :absMax;

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

template<typename T>
__global__ void reduce_minmax_kernel_multi_channels(T * input,T *d_out,size_t size,int numBlocks,T absMin,T absMax, size_t channelCount) {
	int tid = threadIdx.x;                              // Local thread index
	int localX = blockIdx.x * blockDim.x + threadIdx.x;// Global thread index

	extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
	T *minMapTemp = reinterpret_cast<T *>(my_smem);

	// --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
	for (size_t k=0; k<channelCount; k++) {
		minMapTemp[tid+blockDim.x*2*k] = (localX<size) ? input[localX*channelCount+k] : absMin;
		minMapTemp[tid+blockDim.x*(2*k+1)] = (localX<size) ? input[localX*channelCount+k] :absMax;
	}

	// --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

	// --- Reduction in shared memory. Only half of the threads contribute to reduction.
	for (unsigned int s=blockDim.x/2; s>0; s>>=1)
	{
		if (tid < s) {
			for (size_t k=0; k<channelCount; k++) {
				if(minMapTemp[tid+blockDim.x*2*k]<minMapTemp[tid + blockDim.x*2*k + s])
				minMapTemp[tid+blockDim.x*2*k] =minMapTemp[tid + blockDim.x*2*k + s];

				int offset=tid+blockDim.x*(2*k+1);
				if(minMapTemp[offset]>minMapTemp[offset + s])
				minMapTemp[offset] =minMapTemp[offset + s];
			}
		}
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
		__syncthreads();
	}

	if (tid == 0) {
		for (size_t k=0; k<channelCount; k++) {
			d_out[blockIdx.x + blockIdx.x*2*k] = minMapTemp[blockDim.x*2*k];
			d_out[numBlocks + blockIdx.x*(2*k+1)] = minMapTemp[blockDim.x*(2*k+1)];
		}
	}
}

#define NUM_PARTS  1024
#define NUM_BINS 256
template<typename T>
//https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
__global__ void histogram_smem_atomics(T *input, int width, int height,
		unsigned int *out, float min, float ratio) {
	// pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// grid dimensions
	int nx = blockDim.x * gridDim.x;
	int ny = blockDim.y * gridDim.y;

	// linear thread index within 2D block
	int t = threadIdx.x + threadIdx.y * blockDim.x;

	// total threads in 2D block
	int nt = blockDim.x * blockDim.y;

	// linear block index within 2D grid
	int g = blockIdx.x + blockIdx.y * gridDim.x;

	// initialize temporary accumulation array in shared memory
	__shared__
	unsigned int smem[NUM_BINS];

	for (int i = t; i < NUM_BINS; i += nt)
		smem[i] = 0;

	__syncthreads();

	// process pixels
	// updates our block's partial histogram in shared memory
	for (int col = x; col < width; col += nx)
		for (int row = y; row < height; row += ny) {
			T val = input[row * width + col];

			float r = (val - min) * ratio;
			if (r < 0)
				r = 0;
			if (r > 1)
				r = 1;

			unsigned int pos = (unsigned int) (r * (NUM_BINS - 1));
			atomicAdd(&smem[pos], 1);
		}
	__syncthreads();

	// write partial histogram into the global memory
	out += g * NUM_PARTS;
	for (int i = t; i < NUM_BINS; i += nt) {
		out[i] = smem[i];
	}
}


__global__ void histogram_final_accum(const unsigned int *in, int n, unsigned int *out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NUM_BINS) {
    unsigned int total = 0;
    for (int j = 0; j < n; j++)
    {
      total += in[i + NUM_PARTS * j];
    }
    out[i] = total;
  }
}


template<typename T>
__global__ void renderInline_kernel(T * output,T *volumeArray,uint imageW, uint imageH, size_t pos,int width, int height)
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x < imageW) && (y < imageH))
    {
       //output[y*imageW+x]=volumeArray[pos*imageW*imageH+y*imageW+x];
    	output[y*imageW+x]=volumeArray[pos*width*height+y*width+x];

    }
}
template<typename T>
__global__ void renderXline_kernel(T * output,T *volumeArray, uint h, uint d, uint pos,int width, int height) {
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint z = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x < h) && (z < d)) {
		// write output color
		short val = volumeArray[x * width * height
				+ z * width + pos];
		 output[z*h+x]=val;
	}
}


#endif


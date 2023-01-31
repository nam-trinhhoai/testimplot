#ifndef CUDA_VOLUME_KERNEL_CUH_
#define CUDA_VOLUME_KERNEL_CUH_

#include <cuda.h>
#include <cufft.h>
#include <cstdio>

#include "cuda_swap_kernel.cuh"

__global__ void swapAndLoadVolume_kernel(short *volume, short *slice, uint w,
		uint h, uint z) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		volume[z * w * h + x * h + y] = Swap2Bytes(slice[y * w + x]);
	}
}

__global__ void swapAndLoadVolumeTwoVolumes_kernel(short *volume, short *slice,
		short *volume1, short *slice1, uint w, uint h, uint z) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		volume[z * w * h + x * h + y] = Swap2Bytes(slice[y * w + x]);
		volume1[z * w * h + x * h + y] = Swap2Bytes(slice1[y * w + x]);
	}
}

template<typename InputType>
__global__ void swapImage_kernel(InputType *output, InputType *data, uint w, uint h, uint c) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		for (uint k=0; k<c; k++) {
			output[(y * w + x)*c+k] = byteswap(data[(x * h + y)*c+k]);
		}
	}
}

//Iso value extraction
__global__ void isoSurfaceExtract_kernel(short *rgt, short *isoValueSurface,
		uint w, uint h, uint d, unsigned int val) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.y * blockDim.y + threadIdx.y;
	short val1, val2;
	if (x < w && z < d) {
		short y = 0, y1 = 1;
		val1 = rgt[z * w * h + y * w + x];
		val2 = rgt[z * w * h + y1 * w + x];
		while (!(val1 <= val && val2 >= val) && y < h - 1) {
			y++;
			y1++;
			val1 = rgt[z * w * h + y * w + x];
			val2 = rgt[z * w * h + y1 * w + x];
		}
		isoValueSurface[z * w + x] = y;
	}
}

template<typename InputType>
__global__ void isoLineExtract_kernel(InputType *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *out) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (dir == 0) {
		if (x < w) {
			out[2 * x] = x;
			out[2 * x + 1] = isovalueArray[ilPos * w + x];
		}
	} else {
		if (x < h) {
			out[2 * x] = x;
			out[2 * x + 1] = isovalueArray[x * w + ilPos];
		}
	}
}

//RGB spectral decomp
__global__ void cufftComplexModule(cufftComplex *in, float *f1Array,
		float *f2Array, float *f3Array, uint w, uint d, uint height, int f1,
		int f2, int f3) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && z < d) {
		int posBase = z * w * height + x * height;
		f1Array[z * w + x] = sqrt(
				in[posBase + f1].x * in[posBase + f1].x
						+ in[posBase + f1].y * in[posBase + f1].y);
		f2Array[z * w + x] = sqrt(
				in[posBase + f2].x * in[posBase + f2].x
						+ in[posBase + f2].y * in[posBase + f2].y);
		f3Array[z * w + x] = sqrt(
				in[posBase + f3].x * in[posBase + f3].x
						+ in[posBase + f3].y * in[posBase + f3].y);
	}
}

__global__ void cufftComplexModuleAll(cufftComplex *in, float **fArray,
		int offset, uint w, uint d, uint height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && z < d) {
		int posBase = z * w * height + x * height;
		for (int i = 0; i < height; i++)
			fArray[i][offset + z * w + x] = sqrt(
					in[posBase + i].x * in[posBase + i].x
							+ in[posBase + i].y * in[posBase + i].y);
//			fArray[0][offset + z * w + x] = 1000;

	}
}

__global__ void hanningValueBlocExtract_kernel(short *seismic,
		short *isoValueSurface, float *out, uint w, uint h, uint d,
		uint window) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned short window2 = (window - 1) / 2;

	if (x < w && z < d) {
		int index = 0;
		for (int iy = -window2; iy <= window2; iy++) {
			float wt = 0.5f
					* (1.0f
							+ cos(
									6.28318f
											* ((float) index
													/ (float) (window - 1)
													- 0.5f)));

			int y = isoValueSurface[z * w + x] + iy;
			if (y >= 0 && y < h) {
				out[z * w * window + x * window + index] = wt
						* seismic[z * w * h + y * w + x];
			}
			index++;
		}
	}
}
__global__ void meanValueBlocExtract_kernel(short *seismic, short *output,
		short *isoValue, uint w, uint h, uint d, uint window) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned short window2 = (window - 1) / 2;

	if (x < w && z < d) {
		int count = 0;
		float sum = 0;
		for (int iy = -window2; iy <= window2; iy++) {
			int y = isoValue[z * w + x] + iy;
			if (y >= 0 && y < h) {
				sum += seismic[z * w * h + y * w + x];
				count++;
			}
		}
		output[z * w + x] = (short) (sum / count);
	}
}

/*//Test Swaps
__global__ void testSwapKernel(void) {
	// test char
	signed char sCharVal = 105;
	signed char sCharOut = byteswap(sCharVal);
	printf("Signed char in: %d , out : %d \n", (int) sCharVal, (int) sCharOut);

	// test uchar
	unsigned char uCharVal = 105;
	unsigned char uCharOut = byteswap(uCharVal);
	printf("Unsigned char in: %d , out : %d \n", (int) uCharVal, (int) uCharOut);

	// test short
	short shortVal = -70*256+105;
	short shortOut = byteswap(shortVal);
	printf("Signed short in: %d , out : %d \n", (int) shortVal, (int) shortOut);

	// test unsigned short
	unsigned short uShortVal = 186*256+105;
	unsigned short uShortOut = byteswap(uShortVal);
	printf("Unsigned short in: %d , out : %d \n", (int) uShortVal, (int) uShortOut);

	// test int
	int intVal = (-70*256+105) * 256*256 + 186*256+105;
	int intOut = byteswap(intVal);
	printf("Signed int in: %d , out : %d \n", intVal, intOut);

	// test unsigned int
	unsigned int uIntVal = 0x5AA5AA55; //((unsigned int) 186*256+105) * 256*256 + 186*256+105;
	unsigned int uIntOut = byteswap(uIntVal);
	printf("Unsigned int in: %X , out : %X \n", uIntVal, uIntOut);

	float* floatValPtr = (float*)(void*)&intVal;
	float floatOut = byteswap(*floatValPtr);
	int floatAsIntOut = *((int*)((void*) &floatOut));
	printf("Float as int in: %d , out : %d \n", intVal, floatAsIntOut);

	long longVal = 0x0FFFF0005AA5AA55;
	double* doubleValPtr = (double*)(void*)&longVal;
	double doubleOut = byteswap(*doubleValPtr);
	long doubleAsLongOut = *((long*)((void*) &doubleOut));
	printf("Double as long in: %lX , out : %lX \n", longVal, doubleAsLongOut);
}*/

#endif

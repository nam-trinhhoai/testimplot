#include "cuda_optvolume_kernel.cuh"
#include "cuda_common_helpers.h"
#include <limits>

template<typename SeismicType, typename RgtType>
void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,SeismicType *volume, RgtType *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window) {
	const int bloc_size = 32; // 256 threads per block
	dim3 dimBlock(bloc_size, bloc_size);

	dim3 dimGrid((w - 1) / bloc_size + 1, (d - 1) / bloc_size + 1);

	attrAndIsoSurfaceExtractOpt_kernel<<<dimGrid, dimBlock,0,stream>>>(rgt,isovalueArray,volume,attributeArray,w, h, d, val,window);
}

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,signed char *volume, signed char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned char *volume, signed char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,short *volume, signed char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned short *volume, signed char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,int *volume, signed char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned int *volume, signed char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,float *volume, signed char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,double *volume, signed char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,signed char *volume, unsigned char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned char *volume, unsigned char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,short *volume, unsigned char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned short *volume, unsigned char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,int *volume, unsigned char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned int *volume, unsigned char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,float *volume, unsigned char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,double *volume, unsigned char *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,signed char *volume, short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned char *volume, short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,short *volume, short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned short *volume, short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,int *volume, short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned int *volume, short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,float *volume, short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,double *volume, short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,signed char *volume, unsigned short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned char *volume, unsigned short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,short *volume, unsigned short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned short *volume, unsigned short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,int *volume, unsigned short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned int *volume, unsigned short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,float *volume, unsigned short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,double *volume, unsigned short *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,signed char *volume, int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned char *volume, int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,short *volume, int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned short *volume, int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,int *volume, int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned int *volume, int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,float *volume, int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,double *volume, int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,signed char *volume, unsigned int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned char *volume, unsigned int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,short *volume, unsigned int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned short *volume, unsigned int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,int *volume, unsigned int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned int *volume, unsigned int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,float *volume, unsigned int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,double *volume, unsigned int *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,signed char *volume, float*rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned char *volume, float *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,short *volume, float *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned short *volume, float *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,int *volume, float *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned int *volume, float *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,float *volume, float *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,double *volume, float *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,signed char *volume, double *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned char *volume, double *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,short *volume, double *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned short *volume, double *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,int *volume, double *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,unsigned int *volume, double *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,float *volume, double *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template void attributeAndIsoValueBlocExtractOpt(const cudaStream_t & stream,double *volume, double *rgt,
	short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
	uint window);

template<typename InputType>
void computeMinMaxOptimizedOpt(InputType *cuda_image_array,size_t N, float & min,float & max)
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
	cudaMalloc(&tempVector,2*NumBlocks * sizeof(InputType));

	dim3 dimBlock(NumThreads,1);
	dim3 dimGrid(NumBlocks,1);
	reduce_minmax_kernel_opt<<<dimGrid, dimBlock, smemSize>>>(cuda_image_array,(InputType *)tempVector, N,NumBlocks,std::numeric_limits<InputType>::lowest(),std::numeric_limits<InputType>::max());

	InputType res[2*NumBlocks];
	cudaMemcpy((void *) res,tempVector,2*NumBlocks * sizeof(InputType),cudaMemcpyDeviceToHost);

	// --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
	max = std::numeric_limits<InputType>::lowest();
	min = std::numeric_limits<InputType>::max();
	for (int i=0; i<NumBlocks; i++)
	{
		if(max<res[i])
			max =res[i];

		if(min>res[i+NumBlocks])
			min =res[i+NumBlocks];
	}

	cudaFree(tempVector);
}

template void computeMinMaxOptimizedOpt(signed char *cuda_image_array,size_t N, float & min,float & max);
template void computeMinMaxOptimizedOpt(unsigned char *cuda_image_array,size_t N, float & min,float & max);
template void computeMinMaxOptimizedOpt(short *cuda_image_array,size_t N, float & min,float & max);
template void computeMinMaxOptimizedOpt(unsigned short *cuda_image_array,size_t N, float & min,float & max);
template void computeMinMaxOptimizedOpt(int *cuda_image_array,size_t N, float & min,float & max);
template void computeMinMaxOptimizedOpt(unsigned int *cuda_image_array,size_t N, float & min,float & max);
template void computeMinMaxOptimizedOpt(float *cuda_image_array,size_t N, float & min,float & max);
template void computeMinMaxOptimizedOpt(double *cuda_image_array,size_t N, float & min,float & max);

extern "C" void hanningAndIsoBlocExtract(const cudaStream_t &stream,short *seismic, short *rgt, short *isovalueArray,float *data_f,uint w, uint h, uint d, short val,
		uint window) {
	const int bloc_size = 32; // 256 threads per block
	dim3 dimBlock(bloc_size, bloc_size);
	dim3 dimGrid((w - 1) / bloc_size + 1, (d - 1) / bloc_size + 1);

	hanningAndisoSurfaceExtractOpt_kernel<<<dimGrid, dimBlock,0,stream >>>(rgt,isovalueArray,seismic,data_f,w, h, d,val, window);
}


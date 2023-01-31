#include "cuda_volume_kernel.cuh"
#include "cuda_common_helpers.h"

extern "C" short* initCuda3DVolume(short *h_volume,
		const cudaExtent &volumeSize) {
	short *data;
	size_t totalSize = volumeSize.width * volumeSize.height * volumeSize.depth
			* sizeof(short);
	checkCudaErrors(cudaMalloc(&data, totalSize));

	int w = volumeSize.height; //Before transpose!
	int h = volumeSize.width;
	int d = volumeSize.depth;

	void *tempLine;
	checkCudaErrors(cudaMalloc(&tempLine, w * h * sizeof(short)));

	const int bloc_size = 32;
	dim3 dimBlock(bloc_size, bloc_size);
	dim3 dimGrid((w - 1) / bloc_size + 1, (h - 1) / bloc_size + 1);

	for (size_t k = 0; k < d; k++) {
		checkCudaErrors(
				cudaMemcpy(tempLine, h_volume + k * w * h,
						w * h * sizeof(short), cudaMemcpyHostToDevice));
	swapAndLoadVolume_kernel<<<dimGrid, dimBlock>>>(data,(short *)tempLine,w, h, k);
}
cudaFree(tempLine);
return data;
}

extern "C" void blocFFT(short *seismic, short *rgt, short *isovalueArray,
	float *f1Array, float *f2Array, float *f3Array, unsigned int w,
	unsigned int h, unsigned int d, unsigned int val, unsigned int window,
	unsigned int f1, unsigned int f2, unsigned int f3) {

float *data_f;
cudaMalloc(&data_f, w * window * d * sizeof(float));

cufftComplex *deviceOutputData;
cudaMalloc(&deviceOutputData, w * d * (window / 2 + 1) * sizeof(cufftComplex));

const int bloc_size = 32; // 256 threads per block
dim3 dimBlock(bloc_size, bloc_size);
dim3 dimGrid((w - 1) / bloc_size + 1, (d - 1) / bloc_size + 1);

isoSurfaceExtract_kernel<<<dimGrid, dimBlock>>>(rgt,isovalueArray,w, h, d, val);
hanningValueBlocExtract_kernel<<<dimGrid, dimBlock >>>(seismic,isovalueArray,data_f,w, h, d, window);

// --- Batched 1D FFTs
cufftHandle handle;
int rank = 1;                           // --- 1D FFTs
int n[] = { (int) window };                 // --- Size of the Fourier transform
int istride = 1, ostride = 1; // --- Distance between two successive input/output elements
int idist = window, odist = (window / 2 + 1);    // --- Distance between batches
int inembed[] = { 0 };  // --- Input size with pitch (ignored for 1D transforms)
int onembed[] = { 0 }; // --- Output size with pitch (ignored for 1D transforms)
// --- Number of batched executions
cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride,
		odist, CUFFT_R2C, w * d);

cufftExecR2C(handle, data_f, deviceOutputData);

cufftComplexModule<<<dimGrid, dimBlock >>>(deviceOutputData,f1Array,f2Array,f3Array, w,d, window/2+1, f1, f2, f3);

cudaFree(data_f);
cudaFree(deviceOutputData);
cufftDestroy(handle);
}

extern "C" void blocFFTAll(short *seismic, short *rgt, short *isovalueArray,
	float **fArray, unsigned int w, unsigned int h, unsigned int d,
	unsigned int val, unsigned int window) {

	float *data_f;
	cudaMalloc(&data_f, w * window * d * sizeof(float));

	cufftComplex *deviceOutputData;
	cudaMalloc(&deviceOutputData, w * d * (window / 2 + 1) * sizeof(cufftComplex));

	const int bloc_size = 32; // 256 threads per block
	dim3 dimBlock(bloc_size, bloc_size);
	dim3 dimGrid((w - 1) / bloc_size + 1, (d - 1) / bloc_size + 1);

	isoSurfaceExtract_kernel<<<dimGrid, dimBlock>>>(rgt,isovalueArray,w, h, d, val);
	hanningValueBlocExtract_kernel<<<dimGrid, dimBlock >>>(seismic,isovalueArray,data_f,w, h, d, window);

	// --- Batched 1D FFTs
	cufftHandle handle;
	int rank = 1;                           // --- 1D FFTs
	int n[] = { (int) window };                 // --- Size of the Fourier transform
	int istride = 1, ostride = 1; // --- Distance between two successive input/output elements
	int idist = window, odist = (window / 2 + 1);    // --- Distance between batches
	int inembed[] = { 0 };  // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 }; // --- Output size with pitch (ignored for 1D transforms)
	// --- Number of batched executions
	cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride,
			odist, CUFFT_R2C, w * d);

	cufftExecR2C(handle, data_f, deviceOutputData);

	cufftComplexModuleAll<<<dimGrid, dimBlock >>>(deviceOutputData,fArray,0, w,d, window/2+1);

	cudaFree(data_f);
	cudaFree(deviceOutputData);
	cufftDestroy(handle);
}

extern "C" void optimizedFFTModule(const cudaStream_t &stream,
		cufftComplex *deviceOutputData, float *f1Array, float *f2Array,
		float *f3Array, unsigned int w, unsigned int h, unsigned int d,
		unsigned int window, unsigned int f1, unsigned int f2, unsigned int f3) {

	const int bloc_size = 32; // 256 threads per block
	dim3 dimBlock(bloc_size, bloc_size);
	dim3 dimGrid((w - 1) / bloc_size + 1, (d - 1) / bloc_size + 1);

	cufftComplexModule<<<dimGrid, dimBlock,0,stream >>>(deviceOutputData,f1Array,f2Array,f3Array, w,d, window/2+1, f1, f2, f3);
}

extern "C" void optimizedFFTModuleAll(const cudaStream_t &stream,
cufftComplex *deviceOutputData, float **fArray,int offset, unsigned int w, unsigned int h,
unsigned int d, unsigned int window) {

const int bloc_size = 32; // 256 threads per block
dim3 dimBlock(bloc_size, bloc_size);
dim3 dimGrid((w - 1) / bloc_size + 1, (d - 1) / bloc_size + 1);

cufftComplexModuleAll<<<dimGrid, dimBlock,0,stream >>>(deviceOutputData,fArray,offset, w,d, window/2+1);
}

extern "C" void attributeAndIsoValueBlocExtract(short *volume, short *rgt,
short *attributeArray, short *isovalueArray, uint w, uint h, uint d, short val,
uint window) {
const int bloc_size = 32; // 256 threads per block
dim3 dimBlock(bloc_size, bloc_size);
dim3 dimGrid((w - 1) / bloc_size + 1, (d - 1) / bloc_size + 1);

isoSurfaceExtract_kernel<<<dimGrid, dimBlock>>>(rgt,isovalueArray,w, h, d, val);
meanValueBlocExtract_kernel<<<dimGrid, dimBlock>>>(volume,attributeArray,isovalueArray,w, h, d, window);
}

template<typename InputType>
void isoLineExtract(InputType *isovalueArray, uint w, uint h,
unsigned int ilPos, unsigned int dir, float *line) {
const int bloc_size = 32; // 256 threads per block
if (dir == 0) {
dim3 dimBlock(bloc_size);
dim3 dimGrid((w - 1) / bloc_size + 1);

isoLineExtract_kernel<<<dimGrid, dimBlock>>>(isovalueArray,w, h, ilPos, dir,line);
} else {
dim3 dimBlock(bloc_size);
dim3 dimGrid((h - 1) / bloc_size + 1);
isoLineExtract_kernel<<<dimGrid, dimBlock>>>(isovalueArray,w, h, ilPos, dir,line);
}
}

template void isoLineExtract<signed char>(signed char *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *line);
template void isoLineExtract<unsigned char>(unsigned char *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *line);
template void isoLineExtract<short>(short *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *line);
template void isoLineExtract<unsigned short>(unsigned short *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *line);
template void isoLineExtract<int>(int *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *line);
template void isoLineExtract<unsigned int>(unsigned int *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *line);
template void isoLineExtract<float>(float *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *line);
template void isoLineExtract<double>(double *isovalueArray, uint w, uint h,
		unsigned int ilPos, unsigned int dir, float *line);

template<typename InputType>
void byteSwapAndTransposeImageData(InputType *output, InputType *data,
uint w, uint h, uint c) {
const int bloc_size = 32;
dim3 dimBlock(bloc_size, bloc_size);
dim3 dimGrid((w - 1) / bloc_size + 1, (h - 1) / bloc_size + 1);
swapImage_kernel<<<dimGrid, dimBlock>>>(output,data,w, h, c);
}

template void byteSwapAndTransposeImageData(signed char*, signed char*, uint, uint, uint);
template void byteSwapAndTransposeImageData(unsigned char*, unsigned char*, uint, uint, uint);
template void byteSwapAndTransposeImageData(short*, short*, uint, uint, uint);
template void byteSwapAndTransposeImageData(unsigned short*, unsigned short*, uint, uint, uint);
template void byteSwapAndTransposeImageData(int*, int*, uint, uint, uint);
template void byteSwapAndTransposeImageData(unsigned int*, unsigned int*, uint, uint, uint);
template void byteSwapAndTransposeImageData(float*, float*, uint, uint, uint);
template void byteSwapAndTransposeImageData(double*, double*, uint, uint, uint);

/*// Test swaps
void testSwap() {
    testSwapKernel<<<1,1>>>();

    cudaThreadSynchronize();
}*/

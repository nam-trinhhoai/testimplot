#include <cuda.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

#include <sys/sysinfo.h>
#include "sys/types.h"

// #include "RGT_Spectrum_Memory.cuh"

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
extern "C"
int  RGTMemorySpectrum ( cufftReal *hostInputData, short *module, long dimx,long dimy,long dimz, long DATASIZE)
{
    int ndev;
    cudaDeviceProp prop;

    cufftComplex *hostOutputData;

    //Query device properties
    cudaGetDeviceCount(&ndev);
    cudaGetDeviceProperties(&prop,0);
    size_t FreeDeviceMemory, TotalDeviceMemory ;
    cudaMemGetInfo      (&FreeDeviceMemory, &TotalDeviceMemory  ) ;
    size_t  max_z  =  0.7*FreeDeviceMemory /(dimy*DATASIZE * sizeof(cufftReal) + dimy*(DATASIZE / 2 + 1) * sizeof(cufftComplex) +8*dimy*DATASIZE*sizeof(cufftComplex)) ;

    if(max_z > dimz) max_z = dimz ;
    if(max_z <= 0) max_z = 1;

    int nbBloc = dimz/max_z  ;
    int resZ = dimz - nbBloc *max_z ;
    if(resZ > 0 ) nbBloc ++ ;


    float wt ;
    // --- Device side input data allocation and initialization
    cufftReal *deviceInputData;
    gpuErrchk(cudaMalloc((void**)&deviceInputData, max_z*dimy*DATASIZE * sizeof(cufftReal)));

    // --- Host side output data allocation
    gpuErrchk(cudaMallocHost((void**)&hostOutputData, max_z*dimy*(DATASIZE/2 + 1) * sizeof(cufftComplex)));

    // --- Device side output data allocation
    cufftComplex *deviceOutputData;
    gpuErrchk(cudaMalloc((void**)&deviceOutputData, max_z*dimy*(DATASIZE / 2 + 1) * sizeof(cufftComplex)));

    // --- Batched 1D FFTs
    cufftHandle handle;
    int rank = 1;                           // --- 1D FFTs
    int n[] = { DATASIZE };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = DATASIZE, odist = (DATASIZE / 2 + 1); // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    // --- Number of batched executions
    cufftPlanMany(&handle, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_R2C, max_z*dimy);

    for(long numBloc = 0; numBloc < nbBloc ; numBloc ++) {
        int zdeb = numBloc*max_z ;
        if(zdeb + max_z > dimz) {
            max_z = dimz - zdeb ;
        }
        /*
        if ( layerspectrumdialog != nullptr )
        {
            layerspectrumdialog->set_progressbar_values((double)zdeb, (double)(dimz-1));
        }
        */
        printf(" zdeb %d / %d   max_z %d\n",zdeb,dimz,max_z) ;

        size_t indSample = zdeb*dimy*DATASIZE ;
    // ---- Host->Device copy the input
    gpuErrchk(cudaMemcpy(deviceInputData, &(hostInputData[indSample]) , DATASIZE * max_z*dimy * sizeof(cufftReal), cudaMemcpyHostToDevice));

    printf(" NK-2\n") ;
    // ---- Calculate fft
    cufftExecR2C(handle, deviceInputData, deviceOutputData);

    printf(" NK-3\n") ;
    printf("GPU mempry %ld \n", (DATASIZE / 2 + 1) * max_z*dimy * sizeof(cufftComplex)) ;
    // --- Device->Host copy of the results
    gpuErrchk(cudaMemcpy(hostOutputData, deviceOutputData, (DATASIZE / 2 + 1) * max_z*dimy * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    for(long iz=zdeb; iz < zdeb + max_z ; iz ++) {
        for(long iy=0; iy < dimy; iy ++) {

            long ind = ((iz-zdeb)*dimy + iy)*(DATASIZE/2 +1)  ;
            for(long freq = 0; freq < DATASIZE/2 ; freq++) {
                module[(freq +2) * dimz*dimy + iz*dimy + iy] = sqrt(pow(hostOutputData[ind+freq].x, 2) + pow(hostOutputData[ind+freq].y, 2))/(float)(DATASIZE);
            }
        }

    }
    }


    // Write results to file
    cufftDestroy(handle);
    gpuErrchk(cudaFree(deviceOutputData));
    gpuErrchk(cudaFree(deviceInputData));
    gpuErrchk(cudaFreeHost(hostOutputData));
    //gpuErrchk(cudaFree(hostInputData));
    //gpuErrchk(cudaFree(hostOutputData)) ;

    return 0;
}

extern "C"
int  RGTMemorySpectrumBis ( cufftReal *hostInputData, short **module, long dimx,long dimy,long dimz, long DATASIZE)
{
    int ndev;
    cudaDeviceProp prop;

    cufftComplex *hostOutputData;

    //Query device properties
    cudaGetDeviceCount(&ndev);
    cudaGetDeviceProperties(&prop,0);
    size_t FreeDeviceMemory, TotalDeviceMemory ;
    cudaMemGetInfo      (&FreeDeviceMemory, &TotalDeviceMemory  ) ;
    size_t  max_z  =  0.7*FreeDeviceMemory /(dimy*DATASIZE * sizeof(cufftReal) + dimy*(DATASIZE / 2 + 1) * sizeof(cufftComplex) +8*dimy*DATASIZE*sizeof(cufftComplex)) ;

    if(max_z > dimz) max_z = dimz ;
    if(max_z <= 0) max_z = 1;

    int nbBloc = dimz/max_z  ;
    int resZ = dimz - nbBloc *max_z ;
    if(resZ > 0 ) nbBloc ++ ;


    float wt ;
    // --- Device side input data allocation and initialization
    cufftReal *deviceInputData;
    gpuErrchk(cudaMalloc((void**)&deviceInputData, max_z*dimy*DATASIZE * sizeof(cufftReal)));

    // --- Host side output data allocation
    gpuErrchk(cudaMallocHost((void**)&hostOutputData, max_z*dimy*(DATASIZE/2 + 1) * sizeof(cufftComplex)));

    // --- Device side output data allocation
    cufftComplex *deviceOutputData;
    gpuErrchk(cudaMalloc((void**)&deviceOutputData, max_z*dimy*(DATASIZE / 2 + 1) * sizeof(cufftComplex)));

    // --- Batched 1D FFTs
    cufftHandle handle;
    int rank = 1;                           // --- 1D FFTs
    int n[] = { DATASIZE };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = DATASIZE, odist = (DATASIZE / 2 + 1); // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    // --- Number of batched executions
    cufftPlanMany(&handle, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_R2C, max_z*dimy);

    for(long numBloc = 0; numBloc < nbBloc ; numBloc ++) {
        int zdeb = numBloc*max_z ;
        if(zdeb + max_z > dimz) {
            max_z = dimz - zdeb ;
        }
        /*
        if ( layerspectrumdialog != nullptr )
        {
            layerspectrumdialog->set_progressbar_values((double)zdeb, (double)(dimz-1));
        }
        */
        printf(" zdeb %d / %d   max_z %d\n",zdeb,dimz,max_z) ;

        size_t indSample = zdeb*dimy*DATASIZE ;
    // ---- Host->Device copy the input
    gpuErrchk(cudaMemcpy(deviceInputData, &(hostInputData[indSample]) , DATASIZE * max_z*dimy * sizeof(cufftReal), cudaMemcpyHostToDevice));

    printf(" NK-2\n") ;
    // ---- Calculate fft
    cufftExecR2C(handle, deviceInputData, deviceOutputData);

    printf(" NK-3\n") ;
    printf("GPU mempry %ld \n", (DATASIZE / 2 + 1) * max_z*dimy * sizeof(cufftComplex)) ;
    // --- Device->Host copy of the results
    gpuErrchk(cudaMemcpy(hostOutputData, deviceOutputData, (DATASIZE / 2 + 1) * max_z*dimy * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    for(long iz=zdeb; iz < zdeb + max_z ; iz ++) {
        for(long iy=0; iy < dimy; iy ++) {

            long ind = ((iz-zdeb)*dimy + iy)*(DATASIZE/2 +1)  ;
            for(long freq = 0; freq < DATASIZE/2 ; freq++) {
                module[freq+2][iz*dimy + iy] = sqrt(pow(hostOutputData[ind+freq].x, 2) + pow(hostOutputData[ind+freq].y, 2))/(float)(DATASIZE);
            }
        }

    }
    }


    // Write results to file
    cufftDestroy(handle);
    gpuErrchk(cudaFree(deviceOutputData));
    gpuErrchk(cudaFree(deviceInputData));
    gpuErrchk(cudaFreeHost(hostOutputData));
    //gpuErrchk(cudaFree(hostInputData));
    //gpuErrchk(cudaFree(hostOutputData)) ;

    return 0;
}

extern "C"
int RGTMemorySpectrum_getBlocSize (long dimx,long dimy,long dimz, long DATASIZE) {
	int ndev;
    cudaDeviceProp prop;
    
    cudaGetDeviceCount(&ndev);
    cudaGetDeviceProperties(&prop,0);
    size_t FreeDeviceMemory, TotalDeviceMemory ;
    cudaMemGetInfo      (&FreeDeviceMemory, &TotalDeviceMemory  ) ;
    size_t  max_z  =  0.7*FreeDeviceMemory /(dimy*DATASIZE * sizeof(cufftReal) + dimy*(DATASIZE / 2 + 1) * sizeof(cufftComplex) +8*dimy*DATASIZE*sizeof(cufftComplex)) ;

    if(max_z > dimz) max_z = dimz ;
    if(max_z <= 0) max_z = 1;
    
    return max_z;
}


extern "C"
int RGTMemorySpectrum_CPUGPUMemory_getBlocSize (long dimx,long dimy,long dimz, int DATASIZE)
{
	int ndev;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&ndev);
	cudaGetDeviceProperties(&prop,0);
	size_t FreeDeviceMemory, TotalDeviceMemory ;
	cudaMemGetInfo(&FreeDeviceMemory, &TotalDeviceMemory);

	struct sysinfo memInfo;
	sysinfo (&memInfo);
	size_t CPUFreeRam = memInfo.freeram;
	size_t freeRam = FreeDeviceMemory;
	if ( freeRam > CPUFreeRam ) freeRam = CPUFreeRam;

	size_t  max_z  =  0.7*freeRam /(dimy*DATASIZE * sizeof(cufftReal) + dimy*(DATASIZE / 2 + 1) * sizeof(cufftComplex) +8*dimy*DATASIZE*sizeof(cufftComplex)) ;

	if(max_z > dimz) max_z = dimz ;
	if(max_z <= 0) max_z = 1;

	return max_z;
}






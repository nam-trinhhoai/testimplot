#ifndef RGT_SPETCTRUM_MEMORY_CUH
#define RGT_SPETCTRUM_MEMORY_CUH

// #include "LayerSpectrumDialog.h"

#include <cufft.h>
extern "C" int RGTMemorySpectrum ( cufftReal *InputData, short *module,
	long dimx,long dimy,long dimz, long dataSize);
	

extern "C" int RGTMemorySpectrumBis ( cufftReal *InputData, short **module,
	long dimx,long dimy,long dimz, long dataSize);
	
extern "C" int RGTMemorySpectrum_getBlocSize (long dimx,long dimy,long dimz, int dataSize);

extern "C" int RGTMemorySpectrum_CPUGPUMemory_getBlocSize (long dimx,long dimy,long dimz, int dataSize);

	
#endif
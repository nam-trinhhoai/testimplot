#ifndef RGT_MORLET_MEMORY_CUH
#define RGT_MORLET_MEMORY_CUH

#include <complex>

typedef float2 Complex;

extern "C" int  RGTMorletMemory ( Complex *h_signal, short * module, int dimy, int dimz,
	int FILTER_KERNEL_SIZE,int freq_min, int freq_max, int freq_step,
	double sampleRate, int n_cycles);



#endif
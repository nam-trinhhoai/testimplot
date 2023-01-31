#ifndef CUDA_COMMON_HELPERS_H
#define CUDA_COMMON_HELPERS_H


extern "C" void check(unsigned int result, char const *const func, const char *const file, int const line);
#define checkCudaErrors(val) check(static_cast<unsigned int>(val), #val, __FILE__, __LINE__)

extern "C" void __getLastCudaError(const char *errorMessage, const char *file, const int line);
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

typedef unsigned int  uint;

extern "C" unsigned int nextPow2(unsigned int x);

#endif

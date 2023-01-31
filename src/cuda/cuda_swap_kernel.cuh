#define Swap2Bytes(val) \
 ( (((val) >> 8) & 0x00FF) | (((val) << 8) & 0xFF00) )

#define Swap4Bytes(val) \
 ( Swap2Bytes(((val) >> 16)) | ( Swap2Bytes( ((val) & 0x0000FFFF) ) << 16 ) )

#define Swap8Bytes(val) \
 ( Swap4Bytes(((val) >> 32)) | ( Swap4Bytes( ((val) & 0x00000000FFFFFFFF) ) << 32 ) )

template <typename T, size_t n>
struct ByteswapImpl;

template <typename T>
struct ByteswapImpl<T, 1> {
  __device__ T operator()(T& swapIt) const { return swapIt; }
};

template <typename T>
struct ByteswapImpl<T, 2> {
  __device__ T operator()(T& swapIt) const { return Swap2Bytes (swapIt); }
};

template <typename T>
struct ByteswapImpl<T, 4> {
  __device__ T operator()(T& swapIt) const { return Swap4Bytes (swapIt); }
};

template<>
struct ByteswapImpl<float, 4> {
  __device__ float operator()(float& swapIt) const {
    int val = *((int*)&swapIt);
    int out = Swap4Bytes (val);
    float outD = *((float*)&out);
    return outD;
  }
};

template <typename T>
struct ByteswapImpl<T, 8> {
  __device__ T operator()(T& swapIt) const { return Swap8Bytes (swapIt); }
};

template<>
struct ByteswapImpl<double, 8> {
  __device__ double operator()(double& swapIt) const {
    long val = *((long*)&swapIt);
    long out = Swap8Bytes (val);
    double outD = *((double*)&out);
    return outD;
  }
};

template <typename T>
__device__ T byteswap(T& swapIt) { return ByteswapImpl<T, sizeof(T)>()(swapIt); }
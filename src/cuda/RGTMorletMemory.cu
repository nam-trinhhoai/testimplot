#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <complex>

typedef float2 Complex;

static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex*, const Complex*, int, float);
static __device__ __host__ inline Complex ComplexStack(Complex *a, int size) ;
static __global__ void ComplexPointwiseStack(Complex* a,  Complex* c, Complex* kernel, int size, int  kernel_size) ;
Complex ComplexAddHost(Complex a, Complex b) ;
Complex ComplexMulHost(Complex a, Complex b) ;
std::vector<std::complex<double>> compute(double sampleRate, double halfWindow, int n_cycles, double frequency);

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
int  RGTMorletMemory ( Complex *h_signal, short * module, long dimy, long dimz,
	int FILTER_KERNEL_SIZE,int  freq_min,int freq_max, int freq_step,
	double sampleRate, int n_cycles)
{
    // Allocate host memory for the signal
    size_t  h_signal_size=dimy*dimz*FILTER_KERNEL_SIZE ;

    size_t FreeDeviceMemory, TotalDeviceMemory ;
    cudaMemGetInfo      (&FreeDeviceMemory, &TotalDeviceMemory  ) ;
    size_t  max_z  =  (0.3*FreeDeviceMemory - FILTER_KERNEL_SIZE*sizeof(Complex))/(dimy*(1 + FILTER_KERNEL_SIZE)*sizeof(Complex))  ;
    if(max_z > dimz) max_z = dimz ;
    printf(" max_z=%d dimz=%d  \n", max_z, dimz) ;
    int nbBloc = dimz/max_z  ;
    int resZ = dimz - nbBloc *max_z ;
    if(resZ > 0 ) nbBloc ++ ;
    //gpuErrchk(cudaMallocHost((void**)&h_signal,h_signal_size*sizeof(Complex)));
    Complex *h_module ;
    gpuErrchk(cudaMallocHost((void**)&h_module, max_z*dimy*sizeof(Complex)));
    printf(" nk 2 \n") ;
    float frequency ;
    frequency = freq_min ;
    int halfWindow = FILTER_KERNEL_SIZE/2 ;
    // std::vector<std::complex<double>> ws = compute(sampleRate, halfWindow, n_cycles, frequency) ;
    printf(" FILTER_KERNEL_SIZE %d\n",FILTER_KERNEL_SIZE) ;
    // Allocate host memory for the filter
    Complex *h_filter_kernel ;
    gpuErrchk(cudaMallocHost((void**)&h_filter_kernel, FILTER_KERNEL_SIZE*sizeof(Complex)));

    printf(" nk 1 \n") ;
    // Initalize the memory for the filter
    printf(" NK-4\n") ;
    Complex* d_signal;
    gpuErrchk(cudaMalloc((void**)&d_signal, dimy * max_z * FILTER_KERNEL_SIZE* sizeof(Complex)));
    printf(" NK-5\n") ;
    //printf(" NK-1\n") ;
    Complex* d_filter_kernel ;
    gpuErrchk(cudaMalloc((void**)&d_filter_kernel, FILTER_KERNEL_SIZE*sizeof(Complex)));
    printf(" NK-3 \n") ;
    Complex* d_module;
    gpuErrchk(cudaMalloc((void**) &d_module, max_z*dimy*sizeof(Complex))) ;
    gpuErrchk(cudaMemcpy(d_module, h_module, max_z*dimy*sizeof(Complex), cudaMemcpyHostToDevice));
    printf(" NK-2\n") ;
    
        for(int numBloc = 0; numBloc < nbBloc ; numBloc ++) {
        long zdeb = numBloc*max_z ;
        if(zdeb + max_z > dimz) {
            max_z = dimz - zdeb ;
        }
        printf(" zdeb %d / %d   max_z %d\n",zdeb,dimz,max_z) ;
        h_signal_size = max_z*dimy*FILTER_KERNEL_SIZE ;
        size_t  indSample = zdeb*dimy*FILTER_KERNEL_SIZE;
        h_signal_size = max_z*dimy*FILTER_KERNEL_SIZE ;
        printf(" h_signal[indSample].x %f h_signal[indSample].y %f\n", h_signal[indSample].x, h_signal[indSample].y ) ;
        gpuErrchk(cudaMemcpy(d_signal, &(h_signal[indSample]), h_signal_size*sizeof(Complex), cudaMemcpyHostToDevice));


        int nb_freq = 0 ;
        for(int freq=freq_min; freq < freq_max; freq += freq_step) {
            printf(" freq %d / %d\n", freq, freq_max) ;
            frequency = freq ;
            std::vector<std::complex<double>> ws = compute(sampleRate, halfWindow, n_cycles, frequency) ;
            for(int i=0; i < FILTER_KERNEL_SIZE ; i ++ ) {
                //printf(" i %d\n",i) ;
                h_filter_kernel[i].x = ws.at(i).real() ;
                h_filter_kernel[i].y = ws.at(i).imag() ;
                //printf(" i %d ws[%d].x %f ws[%d].y %f\n", i, i,ws.at(i).real(),i, ws.at(i).imag()) ;
                //printf(" i %d ws[%d].x %f ws[%d].y %f\n", i, i,h_filter_kernel[i].x,i, h_filter_kernel[i].y ) ;
            }
            gpuErrchk(cudaMemcpy(d_filter_kernel, h_filter_kernel, FILTER_KERNEL_SIZE*sizeof(Complex),
                                 cudaMemcpyHostToDevice));

            // Copy host memory to device

            dim3 threads(512);
            dim3 nb_bloc((max_z*dimy -1)/threads.x +1) ;
            ComplexPointwiseStack<<<nb_bloc, threads>>>(d_signal,  d_module, d_filter_kernel, max_z*dimy, FILTER_KERNEL_SIZE) ;
            //ComplexPointwiseStack<<<1, 1>>>(d_signal,  d_module, d_filter_kernel, dimz*dimy, FILTER_KERNEL_SIZE) ;

            gpuErrchk(cudaMemcpy(h_module, d_module, max_z*dimy*sizeof(Complex), cudaMemcpyDeviceToHost));
      
            //printf(" update module \n") ;
            for(int i=0; i < max_z*dimy; i++) {
                //if(h_module[i].x != 0.0 ) printf(" h_module[i].x %f h_module[i].y %f\n", h_module[i].x,h_module[i].y) ;
                module[dimz*dimy*(2+nb_freq) + zdeb*dimy + i] =  sqrt(pow(h_module[i].x, 2) + pow(h_module[i].y, 2)) ;

            }
            //printf(" End update module \n") ;
            nb_freq ++ ;
        }
    }
    // cleanup memory

    return (1) ;          
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}
Complex ComplexAddHost(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}
Complex ComplexMulHost(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex stack
static __device__ __host__ inline Complex ComplexStack(Complex *a, int size)
{
    Complex c ;
    int i=0 ;
    c = a[i] ;
    for (int i = 1; i < size; i ++)
        c = ComplexAdd(c, a[i]);

    return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex* a, const Complex* b, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
}
// Complex pointwise multiplication
static __global__ void ComplexPointwiseStack(Complex* a,  Complex* c, Complex* kernel, int size, int  kernel_size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const long threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadID < size) {
        /*
        for (int i = threadID; i < size; i += numThreads) {
        c[i] = ComplexMul(a[i*kernel_size], kernel[0]) ;
        for(int j=1; j < kernel_size; j++) {
           c[i] = ComplexAdd(c[i],ComplexMul(a[i*kernel_size +j], kernel[j]));
        }

        printf("i %d  c[i].x %d c[i].y  %f\n", i,c[i].x ,c[i].y) ;
        */
        c[threadID].x= 0 ;
        c[threadID].y= 0 ;
        for(int j=0; j < kernel_size; j++) {
            c[threadID] = ComplexAdd(c[threadID],ComplexMul(a[threadID*kernel_size +j], kernel[j])) ;
        }
    }
}

int getHalfWindow(double sampleRate, int n_cycles, double frequency) {
    return static_cast<int>(std::ceil(5 * n_cycles / (2 * M_PI * frequency * sampleRate)));
}

std::vector<std::complex<double>> compute(double sampleRate, double halfWindow, int n_cycles, double frequency) {
    double t;
    std::complex<double> W;
    std::vector<std::complex<double>> Ws;
    if (halfWindow<0) {
        halfWindow = getHalfWindow(sampleRate, n_cycles, frequency);
    }
    //printf(" halfWindow %f\n", halfWindow) ;
    Ws.resize(halfWindow*2+1);
    double sigma_t = static_cast<double>(halfWindow) * sampleRate / 5;

    // Compute
    for (int i=0; i<Ws.size(); i++) {
        t = std::fabs(i-halfWindow) * sampleRate;
        W = std::exp(std::complex<double>(0, 2.0 * M_PI * frequency * t));
        Ws[i] = W * std::exp(-std::pow(t, 2) / (2.0 * std::pow(sigma_t, 2)));
    }

    // Compute norm
    double norm = 0;
    for (int i=0; i<Ws.size(); i++) {
        norm += std::pow(Ws[i].real(), 2) + std::pow(Ws[i].imag(), 2);
    } 
    norm = std::sqrt(0.5*norm);

    for (int i=0; i<Ws.size(); i++) {
    }

    return Ws;
}


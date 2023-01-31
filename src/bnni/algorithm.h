#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <vector>

void reflectivityWavelet (float *dt, float *rhob, float pasech, float freq,float *out, int dimx,int wavelet_size);
void reflectivity (float *dt, float *rhob, float pasech, int nplo, int nphi, double flo, double fhi,float *out, int dimx);
void reflectivityFFTW (float *dt, float *rhob, float pasech, float freq,float *out, int dimx);
void SyRicker(float *wavelet,int  npts,float dt,float f00);
void SyConv(float *wavelet,float *sismo,int nw,int ns);

// BandPass  algorithms
void sepBandPass(int nplo, int nphi, double flo, double fhi, double sampleRate, double* in, double* in_tempo, long n1);
std::vector<double>& lowcut(double flo, int nplo, int phase, std::vector<double>& data);
std::vector<double>& highcut(double fhi, int nphi, int phase, std::vector<double>& data);

#endif // ALGORITHM_H

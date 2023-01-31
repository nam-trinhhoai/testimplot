#include "algorithm.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <fftw3.h>
#include <QDebug>

void reflectivityWavelet (float *dt, float *rhob, float pasech, float freq,float *out, int dimx,int wavelet_size)
{
    int i ;
    float *wavelet=NULL;
    float dtime;
    wavelet = (float *)realloc(wavelet,sizeof(float)*wavelet_size);
    dtime = pasech /1000 ;
    printf(" freq %f dt %f\n",freq,dtime) ;
    SyRicker(wavelet,wavelet_size,dtime,freq);
    for (i=0; i < dimx-1 ; i++) {
        if((rhob[i+1]*dt[i+1] + rhob[i]*dt[i]) != 0.0) out[i] = (rhob[i+1]*dt[i+1] - rhob[i]*dt[i])/(rhob[i+1]*dt[i+1] + rhob[i]*dt[i]) ;
    }
    SyConv(wavelet,out,wavelet_size,dimx) ;
}

void reflectivity (float *dt, float *rhob, float pasech, int nplo, int nphi, double flo, double fhi,float *out, int dimx)
{
    int i ;
    //float *wavelet=NULL;
    float dtime;
    //wavelet = (float *)realloc(wavelet,sizeof(float)*wavelet_size);
    dtime = pasech /1000 ;
    printf(" freq %f dt %f\n",fhi,dtime) ;
    //SyRicker(wavelet,wavelet_size,dtime,freq);
    for (i=0; i < dimx-1 ; i++) {
        if((rhob[i+1]*dt[i+1] + rhob[i]*dt[i]) != 0.0) out[i] = (rhob[i+1]*dt[i+1] - rhob[i]*dt[i])/(rhob[i+1]*dt[i+1] + rhob[i]*dt[i]) ;
    }
    double* out_double, *out_double2;
    out_double = (double *)malloc(sizeof(double)*dimx);
    out_double2 = (double *)malloc(sizeof(double)*dimx);
    for (int i=0; i<dimx; i++) {
        out_double[i] = out[i];
    }
    sepBandPass(nplo, nphi, flo, fhi, dtime, out_double, out_double2, dimx) ;
    for (int i=0; i<dimx; i++) {
        out[i] = out_double2[i];
    }
    free(out_double);
    free(out_double2);

    //SyConv(wavelet,out,wavelet_size,dimx) ;
}

void reflectivityFFTW (float *dt, float *rhob, float pasech, float freq,float *out, int dimx) {
	int i ;
	float *wavelet=NULL;
	float dtime;

	long n2 = 1;
	while(n2 < dimx) {
		n2=2*n2 ;
	}

	wavelet = (float *)realloc(wavelet,sizeof(float)*n2);
	dtime = pasech /1000 ;
	printf(" freq %f dt %f\n",freq,dtime) ;
	SyRicker(wavelet,n2,dtime,freq);
	for (i=0; i < dimx-1 ; i++) {
		if((rhob[i+1]*dt[i+1] + rhob[i]*dt[i]) != 0.0) {
			out[i] = (rhob[i+1]*dt[i+1] - rhob[i]*dt[i])/(rhob[i+1]*dt[i+1] + rhob[i]*dt[i]) ;
		} else {
			out[i] = 0;
		}
	}
	out[dimx-1] = 0;

	fftw_complex* S = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n2);
	fftw_complex* M = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n2);
	fftw_complex* SF= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n2);
	fftw_complex* MF= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n2);
	fftw_complex* R = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n2);
	fftw_complex* IR= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n2);
	fftw_plan plan_S = fftw_plan_dft_1d(n2, S , SF, FFTW_FORWARD,  FFTW_ESTIMATE);
	fftw_plan plan_M = fftw_plan_dft_1d(n2, M , MF, FFTW_FORWARD,  FFTW_ESTIMATE);
	fftw_plan plan_R = fftw_plan_dft_1d(n2, R , IR, FFTW_BACKWARD,  FFTW_ESTIMATE);

	for (i = 0; i < n2; i++) {
		S[i][0] = S[i][1] = M[i][1] = 0.0;

		if (i<dimx) {
			S[i][0] = out[i];
		}
		M[i][0] = wavelet[i];
	}
	fftw_execute(plan_S);
	fftw_execute(plan_M);
	for (i = 0; i < n2; i++) {
		R[i][0] = (SF[i][0]*MF[i][0] - SF[i][1]*MF[i][1])/n2;
		R[i][1] = (SF[i][0]*MF[i][1] + SF[i][1]*MF[i][0])/n2;
	}
	fftw_execute(plan_R);
	for (i = 0; i < dimx; i++) {
		// i + n2/2 + (n2/2 - dimx/2)
		// n2 is even
		int index = (i+n2/2) % n2;
		out[i] = IR[index][0];
	}
	//SyConv(wavelet,out,dimx,dimx);

	fftw_destroy_plan(plan_S);
	fftw_destroy_plan(plan_M);
	fftw_destroy_plan(plan_R);
	fftw_free(S);
	fftw_free(SF);
	fftw_free(M);
	fftw_free(MF);
	fftw_free(IR);
	fftw_free(R);

	free(wavelet);
}

void SyRicker(float *wavelet,int  npts,float dt,float f00)
{
    int i, ctre;
    float t;

    ctre = npts/2;
    for(i=0; i<npts; i++) {
        t=(float)(i-ctre)*dt;
        t*=t;
        wavelet[i]=(1.-19.74*f00*f00*t)*exp(-(double)(9.87*f00*f00*t));
    }
}

/*--------------------------------------------------------------
 *         Programme:  SyConv:
 *                 fonction:   Generation de sismo par convolution
 *                 ---------------------------------------------------------------*/

void SyConv(float *wavelet,float *sismo,int nw,int ns)
{
    float som, *out;
    int i,j,m;

    out = (float *)malloc(sizeof(float)*ns);

    m=nw/2;

    for(i=0; i<ns; i++) out[i]=0;

    for(i=0; i<ns; i++) {
        som=0.;
        for(j=0; j<nw; j++) {
            if(i-j+m > 0 && i-j+m < ns)
                som+=(float)sismo[i-j+m]*wavelet[j];
        }
//        if(som > 32000) som = 32000 ;
//        if(som < -32000) som = -32000 ;
        out[i]=som;
    }
    for(i=0; i<ns; i++) sismo[i] = out[i];

    free(out);
}

void sepBandPass(int nplo, int nphi, double flo, double fhi, double sampleRate, double* in, double* in_tempo, long n1) {

    /*if (params.fhi<0 || params.flo<0 || params.plo<=0 || params.phi<=0 || params.sampleRate<=0 ||
            params.fhi<params.flo || x0<0 || step<=0) {
        return std::vector<double>();
    }
    std::vector<double> output((int) ((x1-x0)/step));

    if (params.plo < 1) {
        params.plo = 1;
    }

    if (params.phi < 1) {
        params.phi = 1;
    }


    if (params.fhi == 0) {
        params.fhi = 0.5 / params.sampleRate;
    }

    double x=x0;
    double flo = params.flo * params.sampleRate;
    double fhi = params.fhi * params.sampleRate;
    x0 *= params.sampleRate;
    x1 *= params.sampleRate;
    step *= params.sampleRate;

    double* data = output.data();

    for (int i=0; i<(int) ((x1-x0)/step); i++) {
        x = x0 + i * step;
        double val = 100;
        if (flo>0.0001) {
            double tmp = pow(x/flo, 2*params.plo);
            val *= tmp/(1+tmp);
        }
        if (fhi<0.4999) {
             val *=1/(1+std::pow(x/fhi, 2*params.phi));
        }
        data[i] = val;
    }

    return output;*/
    //int nplo = params.plo;
    //int nphi = params.phi;
    //double fhi = params.fhi;
    //double flo = params.flo;
    double d1 = sampleRate; //params.sampleRate; // Sample Rate in the trace in second
    int phase = 0;

    //int n1 = in.size(); // number of samples in the trace
    int n1pad = 0;
    std::vector<double> new_data;
    std::vector<double> tempdata;
    std::vector<double> data;
    //std::vector<double> in_tempo(n1);

    if (fhi == 0) {
        fhi = 0.5f / d1;
    }

    if (flo >= fhi) {
        qDebug() << "Please specify flo < fhi";
    }

    if (nplo < 1) {
        nplo = 1;
    }

    if (nphi < 1) {
        nphi = 1;
    }

    flo = flo * d1;
    fhi = fhi * d1;

    if (flo < 0.0001 && fhi > 0.4999) {
        memcpy(in_tempo, in, sizeof(double)*n1);
    }

    n1pad = n1 + 2;
    new_data = std::vector<double>(n1pad);
    tempdata = std::vector<double>(n1pad);
    data = std::vector<double>(n1pad);

    for (int i1 = 0; i1 < n1pad; i1++) {
        new_data[i1] = 0.f;
        tempdata[i1] = 0.f;

        if (i1 >= 2) {
            data[i1] = in[i1 - 2];
        }
    }

    // Bandpass data : in
    if (flo > 0.000001) {
        data = lowcut(flo, nplo, phase, data);
    }

    for (int i = 2; i < n1pad; i++) {
        in_tempo[i - 2] = data[i];
    }

    if (fhi < 0.4999999) {
        data = highcut(fhi, nphi, phase, data);
    }

    for (int i = 2; i < n1pad; i++) {
        in_tempo[i - 2] = data[i];
    }

    //return in_tempo;
}

/**
   * Butterworth lowcut (highpass) filter.
   *
   * @param flo
   *          float : cutoff frequency
   * @param nplo
   *          int : number of poles
   * @param phase
   *          int : 0=zero phase 1=min phase
   * @param data
   *          float[] : data to be filtered
   * @return float[] <br>
   *         KEYWORDS bandpass filter butterworth high-pass low-pass<br>
   *         SEE ALSO : Lpfilt Bandpass % end of self documentation<br>
   *         Author - Dave Hale<br>
   *         EDIT HISTORY<br>
   *         9-30-85 stew Convex version<br>
   *         4-30-91 steve Saw version with no AP calls<br>
   *         8-01-91 lin Fix the bug and change the wrong comments<br>
   *         14-09-99 james Back? into C fixed various bugs that had slipped in stripped out highcut/lowcut subroutines<br>
   *         02-03-13 make a real java function Technical reference: Oppenheim, A.V., and Schafer, R.W., 1975,<br>
   *         Digital signal processing, Prentice-Hall Inc.
   */
std::vector<double>& lowcut(double flo, int nplo, int phase, std::vector<double>& data) {
    int nodd = nplo % 2;
    if (phase == 0) {
        if (nodd != 0) nplo = (nplo + 1) / 2;
        else nplo = nplo / 2;
        nodd = nplo % 2;
    }

    int nd = (nplo + 1) / 2;
    std::vector<double> d(nd * 5);
    double fno2 = 0.25f; /* Nyquist frequency over two?? */
    double a = (2.f * std::sin(M_PI * fno2) / std::cos(M_PI * fno2));
    double aa = (std::pow(a, 2.0));
    double aap4 = aa + 4;
    double e = (-std::cos(M_PI * (flo + fno2)) / std::cos(M_PI * (flo - fno2)));
    double ee = (std::pow(e, 2.0));
    double dtheta = (M_PI / nplo); /* angular separation of poles */
    double theta0 = nodd != 0 ? 0.0f : dtheta / 2.0f; /* pole closest to real s axis */

    if (nodd != 0) {
        double b1 = a / (a + 2);
        double b2 = (a - 2) / (a + 2);
        double den = (1.0 - b2 * e);
        d[0] = (b1 * (1. - e) / den);
        d[nd] = -d[0];
        d[2 * nd] = 0.f;
        d[3 * nd] = ((e - b2) / den);
        d[4 * nd] = 0.f;
    }

    for (int j = nodd; j < nd; j++) {
        double c = 4. * a;
        double angle = (theta0 + j * dtheta);
        c *= std::cos(angle);
        double b1 = (aa / (aap4 + c));
        double b2 = ((2. * aa - 8.) / (aap4 + c));
        double b3 = ((aap4 - c) / (aap4 + c));
        double den = (1. - b2 * e + b3 * ee);
        d[j] = (b1 * std::pow((1. - e), 2.) / den);
        d[j + nd] = -2.f * d[j];
        d[j + 2 * nd] = d[j];
        double work1 = e * (1.f + b3);
        work1 = 2.f * work1;
        double work2 = b2 * (1.f + ee);
        d[j + 3 * nd] = (work1 - work2) / den;
        work1 = -b2 * e + b3;
        d[j + 4 * nd] = (ee + work1) / den;
    }

    int n1 = data.size();
    std::vector<double> newdata(n1);
    std::vector<double> tempdata(n1);

    /* lowcut filter */
    for (int ilo = 0; ilo < nd; ilo++) {
        for (int i1 = 2; i1 < n1; i1++) {
            newdata[i1] = d[ilo] * data[i1] + d[ilo + nd] * data[i1 - 1] + d[ilo + 2 * nd] * data[i1 - 2] - d[ilo + 3 * nd] * newdata[i1 - 1] - d[ilo + 4 * nd] * newdata[i1 - 2];
        }

        for (int i1 = 0; i1 < n1; i1++) {
            data[i1] = newdata[i1];
        }
    }

    /* lowcut again in reverse */
    if (phase == 0) {
        for (int i1 = 2; i1 < n1; i1++) {
            tempdata[i1] = data[n1 + 1 - i1];
        }

        for (int ilo = 0; ilo < nd; ilo++) {
            for (int i1 = 2; i1 < n1; i1++)
                newdata[i1] = d[ilo] * tempdata[i1] + d[ilo + nd] * tempdata[i1 - 1] + d[ilo + 2 * nd] * tempdata[i1 - 2] - d[ilo + 3 * nd] * newdata[i1 - 1] - d[ilo + 4 * nd] * newdata[i1 - 2];

            for (int i1 = 0; i1 < n1; i1++)
                tempdata[i1] = newdata[i1];
        }

        for (int i1 = 2; i1 < n1; i1++)
            data[i1] = tempdata[n1 + 1 - i1];
    }

    return data;
}

std::vector<double>& highcut(double fhi, int nphi, int phase, std::vector<double>& data) {
    int nodd = nphi % 2;
    if (phase == 0) {
        if (nodd != 0) nphi = (nphi + 1) / 2;
        else nphi = nphi / 2;
        nodd = nphi % 2;
    }

    int nb = (nphi + 1) / 2;
    std::vector<double> b(nb * 5);

    double a = (2.0f * std::tan(M_PI * fhi)); // radius of poles in s-plane
    double aa = std::pow(a, 2.);
    double aap4 = (aa + 4.0f);
    double dtheta = (M_PI / nphi); // angular separation of poles
    double theta0 = nodd != 0 ? 0.0f : dtheta / 2.0f;// pole closest to real s axis

    if (nodd != 0) {
        b[0] = a / (a + 2.0f);
        b[1 * nb] = b[0];
        b[2 * nb] = 0.0f;
        b[3 * nb] = (a - 2.0f) / (a + 2.0f);
        b[4 * nb] = 0.0f;
    }

    for (int j = nodd; j < nb; j++) {
        double angle = theta0 + j * dtheta;
        double c = 4.0 * a * std::cos(angle);
        b[j] = (aa / (aap4 + c));
        b[j + nb] = 2.0f * b[j];
        b[j + 2 * nb] = b[j];
        b[j + 3 * nb] = ((2.0f * aa - 8.0f) / (aap4 + c));
        b[j + 4 * nb] = ((aap4 - c) / (aap4 + c));
    }

    int n1 = data.size();
    std::vector<double> newdata(n1);
    std::vector<double> tempdata(n1);

    // ArmandSibille : use data begin as new initializer
    if (data.size()>1 && newdata.size()>1) {
        newdata[0] = data[0];
        newdata[1] = data[1];
    }

    for (int ihi = 0; ihi < nb; ihi++) {
        for (int i1 = 2; i1 < n1; i1++)
            newdata[i1] = b[ihi] * data[i1] + b[ihi + nb] * data[i1 - 1] + b[ihi + 2 * nb] * data[i1 - 2] - b[ihi + 3 * nb] * newdata[i1 - 1] - b[ihi + 4 * nb] * newdata[i1 - 2];

        for (int i1 = 0; i1 < n1; i1++)
            data[i1] = newdata[i1];
    }

    if (phase == 0) { // highcut again in reverse
        for (int i1 = 2; i1 < n1; i1++)
            tempdata[i1] = data[n1 + 1 - i1];

        // ArmandSibile : use first set value as initializer
        if (newdata.size()>1 && tempdata.size()>2) {
            newdata[0] = tempdata[2];
            newdata[1] = tempdata[2];
        }

      for (int ihi = 0; ihi < nb; ihi++) {
          for (int i1 = 2; i1 < n1; i1++)
              newdata[i1] = b[ihi] * tempdata[i1] + b[ihi + nb] * tempdata[i1 - 1] + b[ihi + 2 * nb] * tempdata[i1 - 2] - b[ihi + 3 * nb] * newdata[i1 - 1] - b[ihi + 4 * nb] * newdata[i1 - 2];

          for (int i1 = 0; i1 < n1; i1++)
              tempdata[i1] = newdata[i1];
      }

      for (int i1 = 2; i1 < n1; i1++)
          data[i1] = tempdata[n1 + 1 - i1];
    }

    return data;

}

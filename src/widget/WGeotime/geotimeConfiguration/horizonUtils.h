

#ifndef __HORIZONUTILS__
#define __HORIZONUTILS__

#include <string>

bool horizonRead(std::string filename, int dimx, int dimy, int dimz,
		float pasech, float tdeb, float **horizon);

float horizonMean(float *horizon, int dimy, int dimz);

#endif

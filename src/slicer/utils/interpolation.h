#ifndef SRC_SLICER_UTILS_INTERPOLATION_H_
#define SRC_SLICER_UTILS_INTERPOLATION_H_

#include "resamplespline.h"

#include <vector>

//#define EPSILON_INTERPOLATION 1.0e-3

typedef struct BilinearPoint {
	double i, j, w;
} BilinearPoint;

// define point i,j and grid iMin->iMax, jMin->jMax with grid step iStep, jStep
std::vector<BilinearPoint> bilinearInterpolationPoints(double i, double j, double iMin,
		double jMin, double iMax, double jMax, double iStep, double jStep);

void resampleSpline(double newSampleRate, double oldSampleRate, const std::vector<double>& inputData,
		std::vector<double>& outputData);

#endif

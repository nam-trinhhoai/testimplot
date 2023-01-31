#include "interpolation.h"

#include <cmath>

std::vector<BilinearPoint> bilinearInterpolationPoints(double i, double j, double iMin,
		double jMin, double iMax, double jMax, double _iStep, double _jStep) {
	double iStep = std::fabs(_iStep);
	double jStep = std::fabs(_jStep);
	double iEpsilon = std::max(1.0e-30, iStep/1000);
	double jEpsilon = std::max(1.0e-30, jStep/1000);

	std::vector<BilinearPoint> out, outFiltered;

	double i1 = std::floor ((i-iMin) / iStep) * iStep + iMin;
	double i2 = i1 + iStep;// * ((i-i1>=0) ? 1 : -1);
	double j1 = std::floor ((j-jMin) / jStep) * jStep + jMin;
	double j2 = j1 + jStep;// * ((j-j1>=0) ? 1 : -1);

	bool iFound = false;
	double iSet;
	if (std::fabs(i1-i)<iEpsilon) {
		iFound = true;
		iSet = i1;
	} else if (std::fabs(i2-i)<iEpsilon) {
		iFound = true;
		iSet = i2;
	}

	bool jFound = false;
	double jSet;
	if (std::fabs(j1-j)<jEpsilon) {
		jFound = true;
		jSet = j1;
	} else if (std::fabs(j2-j)<jEpsilon) {
		jFound = true;
		jSet = j2;
	}

	if (iFound && jFound) {
		BilinearPoint point;
		point.i = iSet;
		point.j = jSet;
		point.w = 1.0;
		out.push_back(point);
	} else if (iFound) {
		BilinearPoint point1, point2;
		point1.i = iSet;
		point1.j = j1;

		double dist1 = std::fabs(j-j1);
		point1.w = 1.0/dist1;
		out.push_back(point1);

		point2.i = iSet;
		point2.j = j2;

		double dist2 = std::fabs(j-j2);
		point2.w = 1.0/dist2;
		out.push_back(point2);
	} else if (jFound) {
		BilinearPoint point1, point2;
		point1.i = i1;
		point1.j = jSet;

		double dist1 = std::fabs(i-i1);
		point1.w = 1.0/dist1;
		out.push_back(point1);

		point2.i = i2;
		point2.j = jSet;

		double dist2 = std::fabs(i-i2);
		point2.w = 1.0/dist2;
		out.push_back(point2);
	} else {
		BilinearPoint point1, point2, point3, point4;
		point1.i = i1;
		point1.j = j1;

		double dist1 = std::sqrt(std::pow(i-i1, 2)+ std::pow(j-j1, 2));
		point1.w = 1.0/dist1;
		out.push_back(point1);

		point2.i = i2;
		point2.j = j1;

		double dist2 = std::sqrt(std::pow(i-i2, 2)+ std::pow(j-j1, 2));
		point2.w = 1.0/dist2;
		out.push_back(point2);

		point3.i = i1;
		point3.j = j2;

		double dist3 = std::sqrt(std::pow(i-i1, 2)+ std::pow(j-j2, 2));
		point3.w = 1.0/dist3;
		out.push_back(point3);

		point4.i = i2;
		point4.j = j2;

		double dist4 = std::sqrt(std::pow(i-i2, 2)+ std::pow(j-j2, 2));
		point4.w = 1.0/dist4;
		out.push_back(point4);
	}

	for (BilinearPoint pt : out) {
		if (pt.i>=iMin && pt.i<=iMax && pt.j>=jMin && pt.j<=jMax) {
			outFiltered.push_back(pt);
		}
	}

	return outFiltered;
}

void resampleSpline(double newSampleRate, double oldSampleRate, const std::vector<double>& inputData,
		std::vector<double>& outputData) {
	resampleSpline(newSampleRate, oldSampleRate, inputData.data(), inputData.size(), outputData.data(), outputData.size());
}

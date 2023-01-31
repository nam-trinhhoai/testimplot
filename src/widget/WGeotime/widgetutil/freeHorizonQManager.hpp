#ifndef SRC_WIDGET_WGEOTIME_WIDGETUTIL_FREEHORIZONQMANAGER_HPP
#define SRC_WIDGET_WGEOTIME_WIDGETUTIL_FREEHORIZONQMANAGER_HPP

#include "interpolation.h"

#include <iterator>
#include <omp.h>

template<typename InputIterator>
bool FreeHorizonQManager::getCroppedBufferFromGrids(const Grid2D& inputGrid, const Grid2D& outputGrid, InputIterator inputBufBegin,
		InputIterator inputBufEnd, std::vector<float>& outputBuffer) {
	long N = outputGrid.countInline()*outputGrid.countXLine();
	if (N<=0) {
		return false;
	}
	outputBuffer.resize(N);

	bool success = getCroppedBufferFromGrids(inputGrid, outputGrid, inputBufBegin, inputBufEnd, outputBuffer.begin(), outputBuffer.end());
	return success;
}

template<typename InputIterator, typename OutputIterator>
bool FreeHorizonQManager::getCroppedBufferFromGrids(const Grid2D& inputGrid, const Grid2D& outputGrid, InputIterator inputBufBegin,
		InputIterator inputBufEnd, OutputIterator outputBufBegin, OutputIterator outputBufEnd) {
	// need valid grids
	if (!inputGrid.isGridValid() || !outputGrid.isGridValid() || inputGrid.depthAxis()!=outputGrid.depthAxis()) {
		return false;
	}
	// check array sizes
	long Ninput = inputGrid.countInline()*inputGrid.countXLine();
	if (Ninput!=std::distance(inputBufBegin, inputBufEnd)) {
		return false;
	}
	long Noutput = outputGrid.countInline()*outputGrid.countXLine();
	if (Noutput!=std::distance(outputBufBegin, outputBufEnd)) {
		return false;
	}

	double iMin = inputGrid.startXLine();
	double iStep = inputGrid.stepXLine();
	double iMax = iMin + iStep * (inputGrid.countXLine() - 1);
	double jMin = inputGrid.startInline();
	double jStep = inputGrid.stepInline();
	double jMax = jMin + jStep * (inputGrid.countInline() - 1);

	#pragma omp parallel for
	for (long iy=0; iy<outputGrid.countInline(); iy++) {
		double j = iy * outputGrid.stepInline() + outputGrid.startInline();
		for (long ix=0; ix<outputGrid.countXLine(); ix++) {
			double i = ix * outputGrid.stepXLine() + outputGrid.startXLine();
			std::vector<BilinearPoint> points = bilinearInterpolationPoints(i, j, iMin,
					jMin, iMax, jMax, iStep, jStep);

			double val = 0;
			double cumulWeight = 0;
			for (int ptIdx=0; ptIdx<points.size(); ptIdx++) {
				const BilinearPoint& point = points[ptIdx];
				long ixIn = (point.i - iMin) / iStep;
				long iyIn = (point.j - jMin) / jStep;
				if (ixIn>=0 && ixIn<inputGrid.countXLine() && iyIn>=0 && iyIn<inputGrid.countInline()) {
					long inputBufIdx = ixIn + iyIn * inputGrid.countXLine();
					InputIterator it = inputBufBegin;
					std::advance(it, inputBufIdx);
					val += point.w * (*it);
					cumulWeight += point.w;
				}
			}
			if (cumulWeight==0) {
				val = -9999.0; // default null value for horizons
			} else {
				val /= cumulWeight;
			}

			long outputBufIdx = ix + iy * outputGrid.countXLine();
			OutputIterator itOut = outputBufBegin;
			std::advance(itOut, outputBufIdx);
			*itOut = val;
		}
	}
	return true;
}

#endif // SRC_WIDGET_WGEOTIME_WIDGETUTIL_FREEHORIZONQMANAGER_HPP

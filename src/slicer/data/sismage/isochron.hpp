#ifndef SRC_SLICER_DATA_SISMAGE_ISOCHRON_HPP
#define SRC_SLICER_DATA_SISMAGE_ISOCHRON_HPP

#include "interpolation.h"

template<typename OutputType>
void Isochron::interpolateBuffer(OutputType* horizonBuf, long inlineDsDim, long inlineSvDim, long xlineDsDim, long xlineSvDim,
		long firstDsInline, long firstSvInline, long inlineSvStep, long inlineStepFact, long firstDsXline, long firstSvXline,
		long xlineSvStep, long xlineStepFact, float nullValue) {
	size_t svIMin = (firstDsInline - firstSvInline) / inlineSvStep;
	size_t svIMax = (firstDsInline - firstSvInline) / inlineSvStep + (inlineDsDim-1) * inlineStepFact;
	size_t svXMin = (firstDsXline - firstSvXline) / xlineSvStep;
	size_t svXMax = (firstDsXline - firstSvXline) / xlineSvStep + (xlineDsDim-1) * xlineStepFact;

#pragma omp parallel for
	for ( int i = svIMin; i <= svIMax; i++) {
		for ( int x = svXMin; x <= svXMax; x++) {
			if (i>=0 && i<inlineSvDim && x>=0 && x<xlineSvDim && horizonBuf[i * xlineSvDim + x] == nullValue) {
				std::vector<BilinearPoint> interpPoints =  bilinearInterpolationPoints(i, x, svIMin, svXMin, svIMax, svXMax,
							inlineStepFact, xlineStepFact);
				float w = 0;
				float sum = 0;

				for (const BilinearPoint& pt : interpPoints) { // pt.i == i, pt.j == x
					OutputType val = horizonBuf[((long)pt.i) * xlineSvDim + ((long) pt.j)];
					if (val != nullValue) {
						w += pt.w;
						sum += pt.w * val;
					}
				}
				if (w!=0) {
					horizonBuf[i * xlineSvDim + x] = sum/w;
				}
			}
		}
	}
}

#endif // SRC_SLICER_DATA_SISMAGE_ISOCHRON_HPP

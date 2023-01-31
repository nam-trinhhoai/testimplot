#ifndef SRC_SLICER_UTILS_IOUTIL_HPP
#define SRC_SLICER_UTILS_IOUTIL_HPP

#include <cmath>
#include <iterator>

#include <omp.h>


template<typename InputIterator, typename OutputIterator>
bool brick2DToContinuousData(InputIterator inputFirst, InputIterator inputEnd, OutputIterator outputFirst,
		OutputIterator outputEnd, long brickSizeX, long brickSizeY, long sizeX, long sizeY) {
	long N = sizeX * sizeY;
	bool valid = std::distance(inputFirst, inputEnd)==N && std::distance(outputFirst, outputEnd)==N;
	if (!valid) {
		return valid;
	}

	long numBrickY = std::ceil(((double) sizeY)/brickSizeY);
	long numBrickX = std::ceil(((double) sizeX)/brickSizeX);

	#pragma omp parallel for schedule(static)
	for (long iy=0; iy<numBrickY; iy++) {
		long idxL = iy*sizeX*brickSizeY;
		long iby_0 = 64*iy;
		long iby_1 = std::min(iby_0+brickSizeY, sizeY);
		for (long ix=0; ix<numBrickX; ix++) {
			long ibx_0 = 64*ix;
			long ibx_1 = std::min(ibx_0+brickSizeX, sizeX);

			for (long j=iby_0; j<iby_1; j++) {
				for (long i=ibx_0; i<ibx_1; i++) {
					long indexInBlock = (i-ibx_0) + (j-iby_0) * (ibx_1-ibx_0);

					InputIterator inputIt = inputFirst;
					std::advance(inputIt, idxL + indexInBlock);
					OutputIterator outputIt = outputFirst;
					std::advance(outputIt, i+j*sizeX);
					*outputIt = *inputIt;
				}
			}

			idxL += (ibx_1-ibx_0)*(iby_1-iby_0);
		}
	}

	return valid;
}

#endif // SRC_SLICER_UTILS_IOUTIL_HPP

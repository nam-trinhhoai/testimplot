/*
 * Isochron.h
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_DATA_ISOCHRON_H_
#define TARUMAPP_SRC_DATA_ISOCHRON_H_

#include <string>

#include "layer.h"
#include "viewutils.h"
#include "cubeseismicaddon.h"

class QRect;
class LayerSlice;
class Seismic3DAbstractDataset;
class CPUImagePaletteHolder;
class CUDAImagePaletteHolder;
class Seismic3DDataset;

class Isochron {
public:
	Isochron( const std::string& horizonName, const std::string& surveyPath);
	virtual ~Isochron();

	std::string horizonName() const;
	std::string surveyPath() const;

	void saveInto(CUDAImagePaletteHolder *cudaIso, const CubeSeismicAddon& seismicAddon, bool interpolate=false);
	void saveInto(CPUImagePaletteHolder *cpuIso, const CubeSeismicAddon& seismicAddon, bool interpolate=false);

	static std::pair<double, double> getStepFact(const std::string& surveyPath, CubeSeismicAddon seismicAddon);

	static bool isXtHorizon(const std::string& horizonPath);
	static SampleUnit getSampleUnit(const std::string& horizonPath);

	static int readIntFromHeader(FILE* pFile, const char* filter, std::size_t header_size, bool* ok=nullptr);
	static std::string readStrFromHeader(FILE* pFile, const char* filter, std::size_t header_size, bool* ok=nullptr);

	template<typename OutputType>
	static void interpolateBuffer(OutputType* horizonBuf, long inlineDsDim, long inlineSvDim, long xlineDsDim, long xlineSvDim,
			long firstDsInline, long firstSvInline, long inlineSvStep, long inlineStepFact, long firstDsXline, long firstSvXline,
			long xlineSvStep, long xlineStepFact, float nullValue);

private:

	bool writePropIsochron(QRect& sourceRegion,
			short* shortTab, float* floatTab, int slice, std::string& propName,
			SampleUnit sampleUnit,
			double firstSample, double sampleStep,
			double firstSvInline, double inlineSvDim, double inlineSvStep,
			double firstSvXline, double xlineSvDim, double xlineSvStep,
			double firstDsInline, double inlineDsDim, double inlineDsStep,
			double firstDsXline, double xlineDsDim, double xlineDsStep, size_t mapLengthSurvey, bool interpolate);

	std::string m_isochronPath;
	std::string m_horizonName;

	std::string m_surveyPath;
};

#include "isochron.hpp"

#endif /* TARUMAPP_SRC_DATA_ISOCHRON_H_ */

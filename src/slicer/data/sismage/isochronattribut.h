/*
 * isochronattribut.h
 *
 *  Created on: Jan 10, 2023
 *      Author: l0483271
 */

#ifndef SRC_DATA_SISMAGE_ISOCHRONATTRIBUT_H_
#define SRC_DATA_SISMAGE_ISOCHRONATTRIBUT_H_

#include "isochron.h"

class IsochronAttribut {
public:
	IsochronAttribut(const Isochron& isochron, const std::string& attributName);
	virtual ~IsochronAttribut();

	void saveInto(CPUImagePaletteHolder *cpuIso, const CubeSeismicAddon& seismicAddon, bool interpolate=false);

private:
	bool writePropIsochron(QRect& sourceRegion,
			short* attrTab, short* surveyTab, int slice, std::string& propName,
			SampleUnit sampleUnit,
			double firstSample, double sampleStep,
			double firstSvInline, double inlineSvDim, double inlineSvStep,
			double firstSvXline, double xlineSvDim, double xlineSvStep,
			double firstDsInline, double inlineDsDim, double inlineDsStep,
			double firstDsXline, double xlineDsDim, double xlineDsStep, size_t mapLengthSurvey, bool interpolate);

	const Isochron& m_isochron;
	std::string m_attributName;
	std::string m_attributPath;
};

#endif /* SRC_DATA_SISMAGE_ISOCHRONATTRIBUT_H_ */

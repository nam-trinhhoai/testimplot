/*
 * Layering.h
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_DATA_LAYERING_H_
#define TARUMAPP_SRC_DATA_LAYERING_H_

#include <string>

#include "layer.h"

class QRect;
class LayerSlice;


class Layering {
public:
	Layering(const std::string& dirName);
	virtual ~Layering();

	void saveInto(LayerSlice* m_data);

	const std::string& getCategory() const {
		return m_category;
	}

	const std::string& getName() const {
		return m_label;
	}

	int getNbLayer() const {
		return m_nbLayer;
	}

	const std::string& getType() const {
		return m_type;
	}

	const std::string& getDirName() const {
		return m_dirName;
	}

private:
	bool init();
	bool readLayeringDesc(const std::string& filename);
	bool writeProp(/*io::SampleTypeBinder& binder, */LayerSlice* data, QRect& sourceRegion,
			float* oriTab, float* floatTab, int slice, std::string& propName,
			double firstSvInline, double inlineSvDim, double inlineSvStep,
			double firstSvXline, double xlineSvDim, double xlineSvStep,
			double firstDsInline, double inlineDsDim, double inlineDsStep,
			double firstDsXline, double xlineDsDim, double xlineDsStep, size_t mapLengthSurvey);
	bool writePropIsochron(/*io::SampleTypeBinder& binder, */LayerSlice* data, QRect& sourceRegion,
			float* oriTab, float* floatTab, int slice, std::string& propName,
			double firstSample, double sampleStep,
			double firstSvInline, double inlineSvDim, double inlineSvStep,
			double firstSvXline, double xlineSvDim, double xlineSvStep,
			double firstDsInline, double inlineDsDim, double inlineDsStep,
			double firstDsXline, double xlineDsDim, double xlineDsStep, size_t mapLengthSurvey);

	std::string m_dirName;

	bool m_layersInitialized = false;
	std::string m_monoLayerPath;
	std::string m_label;
	std::string m_type;
	std::string m_category;
	int m_nbLayer;

	Layer m_monoLayer;
};

#endif /* TARUMAPP_SRC_DATA_LAYERING_H_ */

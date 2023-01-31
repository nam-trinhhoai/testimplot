/*
 * Layering.h
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_DATA_LAYER_H_
#define TARUMAPP_SRC_DATA_LAYER_H_

#include <string>
#include <vector>

class LayerProperty;

class Layer {
public:
	Layer();
	virtual ~Layer();

	const std::string& getName() const {
		return m_label;
	}

	bool setLayerDirectory(const std::string& dirName);
	bool writeProperty(float *tab, std::string& propName);

	const std::string& getDataFile() const {
		return m_dataFile;
	}

	int getNbProfiles() const {
		return m_nbProfiles;
	}

	int getNbTraces() const {
		return m_nbTraces;
	}

	void setNbTraces(int nbTraces = 1) {
		m_nbTraces = nbTraces;
	}

	int getProfile0() const {
		return m_trace0;
	}

	int getTrace0() const {
		return m_profile0;
	}

private:
	bool readLayeringDesc(const std::string& filename);
	bool readLayerDesc(const std::string& filename);

	std::string m_dirName;
	std::string m_label;
	std::string m_kind;
	int m_trace0 = 0;
	int m_profile0 = 0;
	int m_nbTraces = 1;
	int m_nbProfiles = 1;
	std::string m_dataFile;
	std::string m_descFile;

	std::vector<LayerProperty*> m_layerProperties;
};

#endif /* TARUMAPP_SRC_DATA_LAYER_H_ */

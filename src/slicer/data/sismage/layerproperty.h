/*
 * LayerPropertying.h
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_DATA_LayerProperty_H_
#define TARUMAPP_SRC_DATA_LayerProperty_H_

#include <string>

class LayerProperty {
public:
	LayerProperty(const std::string& dirName);
	virtual ~LayerProperty();

	const std::string& getName() const {
		return m_name;
	}

	bool writeProperty(float *tab, const std::string& propName);

	const std::string& getDataFile() const {
		return m_dataFile;
	}

	int getNbProfiles() const {
		return m_nj;
	}

	int getNbTraces() const {
		return m_ni;
	}

	void setNbTraces(int nbTraces = 1) {
		m_ni = nbTraces;
	}

	int getProfile0() const {
		return m_j0;
	}

	int getTrace0() const {
		return m_i0;
	}

	void setDims( int dimW, int dimH );

private:
	bool readLayerPropertyingDesc(const std::string& filename);
	bool readLayerPropertyDesc(const std::string& filename);
	void headerFromV2ToV1();

	std::string m_dirName;
	std::string m_name;
	std::string m_kind;
	std::string m_originGeometries;
	std::string m_nullValue = "-9999";
	int m_storageVersion = 1;
	int m_j0 = 0;
	int m_i0 = 0;
	int m_ni = 1;
	int m_nj = 1;
	std::string m_dataFile;
	std::string m_descFile;
};


#endif /* TARUMAPP_SRC_DATA_LayerProperty_H_ */

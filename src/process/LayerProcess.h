#ifndef MURATPROCESSLIB_SRC_LayerProcess_H_
#define MURATPROCESSLIB_SRC_LayerProcess_H_

#include <QObject>

#include <vector>
#include <math.h>
#include <cmath>
#include <complex>
#include <memory>
#include <QRect>
#include <QMutex>
#include <QMutexLocker>

#include "LayerSpectrumDialog.h"

class Seismic3DDataset;

class LayerProcess : public QObject {
	Q_OBJECT
public:
	LayerProcess(Seismic3DDataset *datasetS, int channelS, Seismic3DDataset *datasetT, int channelT);

	virtual ~LayerProcess();// {
//		if (m_module!=nullptr) {
//			delete m_module;
//		}
//	};
	virtual void init() {};

	int getNbOutputSlices() const {
		QMutexLocker lock(&m_cacheMutex);
		return m_nbOutputSlices;
	}

	bool isModuleComputed() const {
		QMutexLocker lock(&m_computeMutex);
		return m_isComputed;
	}

//	int getDimW() const {
//		return m_dimW;
//	}
//
//	int getDimH() const {
//		return m_dimH;
//	}

	Seismic3DDataset * getCubeS(	) {
		return m_cubeS;
	}

	Seismic3DDataset * getCubeT() {
		return m_cubeT;
	}

	const Seismic3DDataset * getCubeS(	) const {
		return m_cubeS;
	}

	int getChannelS() const {
		return m_channelS;
	}

	const Seismic3DDataset * getCubeT() const {
		return m_cubeT;
	}

	int getChannelT() const {
		return m_channelT;
	}

	virtual void compute(LayerSpectrumDialog *layerspectrumdialog) = 0;

	/**
	 * Give a pointer to the cached result for the specified spectrumSlice
	 *
	 * ! 0 is used for rgt index
	 * ! 1 is used for cubeS values
	 * ! 2 -> 1 + window/2 is the process result
	 *
	 * The pointer given will be invalid if spectrumSlice > getNbOutputSlices()
	 */
	virtual const float* getModuleData(std::size_t spectrumSlice) const = 0;


	//void readRGBA(const QRect& sourceRegion, int rgbComponent, unsigned char* f3 );
	//void readGray(const QRect& sourceRegion, int spectrumSlice, unsigned char* f3 );

signals:
	void processCacheIsReset();

protected:

	mutable int m_nbOutputSlices = 2 + 64/2;
	mutable bool m_isComputed = false;

	mutable std::vector<std::vector<float>>* m_module = nullptr;

	size_t m_dimW = 1, m_dimH = 1;

	Seismic3DDataset* m_cubeS;
	int m_channelS;
	Seismic3DDataset* m_cubeT;
	int m_channelT;

	mutable QMutex m_cacheMutex; // mutex to use to use cache
	mutable QMutex m_computeMutex; // mutex to use related to computation
};

#endif /* MURATPROCESSLIB_SRC_LayerProcess_H_ */


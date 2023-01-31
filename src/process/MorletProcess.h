/*

 *
 *  Created on: 5 janv. 2018
 *      Author: Georges
 */

#ifndef MURATPROCESSLIB_SRC_MorletProcess_H_
#define MURATPROCESSLIB_SRC_MorletProcess_H_

#include <LayerProcess.h>

#include <vector>
#include <math.h>
#include <cmath>
#include <complex>
#include <memory>
#include <QRect>
#include <QMutex>
#include <QMutexLocker>
#include <QObject>

#include "LayerSpectrumDialog.h"
#include "process/RgtLayerProcessUtil.h"
#include "seismic3ddataset.h"
#include "slicer/data/sliceutils.h"

class ToAnalyse3Process : public LayerProcess {

public:
	ToAnalyse3Process(Seismic3DDataset* cubeS, int channelS,
			Seismic3DDataset* cubeT, int channelT): LayerProcess(cubeS, channelS, cubeT, channelT){};
	virtual ~ToAnalyse3Process() {};
	virtual void init() {};

	/*void setGeologicalTime(int gt) {
		if ( m_geologicalTime == gt )
			return;

		{
			QMutexLocker lock(&m_computeMutex);
			m_geologicalTime = gt;
			m_isComputed = false;
		}
	}*/

	void setSeeds(const std::vector<RgtSeed>& seeds) {
		QMutexLocker lock(&m_computeMutex);
		m_seeds = seeds;
		m_isComputed = false;
	}

	void setDistancePower(int dist) {
		QMutexLocker lock(&m_computeMutex);
		m_distancePower = dist;
		m_isComputed = false;
	}

	void setPolarity(bool pol) {
		if ( m_polarity == pol )
			return;

		{
			QMutexLocker lock(&m_computeMutex);
			m_polarity = pol;
			m_isComputed = false;
		}
	}

	void setUseSnap(bool pol) {
		if ( m_useSnap == pol )
			return;

		{
			QMutexLocker lock(&m_computeMutex);
			m_useSnap = pol;
			m_isComputed = false;
		}
	}

	void setSnapWindow(int val) {
		if (m_snapWindow == val) {
			return;
		}
		{
			QMutexLocker lock(&m_computeMutex);
			m_snapWindow = val;
			m_isComputed = false;
		}
	}

	void setUseMedian(bool pol) {
		if ( m_useMedian == pol )
			return;

		{
			QMutexLocker lock(&m_computeMutex);
			m_useMedian = pol;
			m_isComputed = false;
		}
	}

	void setLWXMedianFilter(int lwx) {
		if ( m_lwx_medianFilter == lwx )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_lwx_medianFilter = lwx;
			m_isComputed = false;
		}
	}

	void setConstrainLayer(const std::vector<float>& constrain) {
        QMutexLocker lock(&m_computeMutex);
        m_constrainIso = constrain;
        m_isComputed = false;
	}

    void setReferenceLayer(const std::vector<ReferenceDuo>& reference) {
        QMutexLocker lock(&m_computeMutex);
        m_reference = reference;
        m_isComputed = false;
    }

	int getNbOutputSlices() const {
		QMutexLocker lock(&m_cacheMutex);
		return m_nbOutputSlices;
	}

	void compute(LayerSpectrumDialog *layerspectrumdialog) override {};

	/**
	 * Give a pointer to the cached result for the specified spectrumSlice
	 *
	 * ! 0 is used for rgt index
	 * ! 1 is used for cubeS values
	 * ! 2 -> 1 + window/2 is the process result
	 *
	 * The pointer given will be invalid if spectrumSlice > getNbOutputSlices()
	 */
	const float* getModuleData(std::size_t spectrumSlice) const override{
		return nullptr;
	};

	void setFreqMax(int freqMax) {
		if ( m_freqMax == freqMax )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_freqMax = freqMax;
			m_isComputed = false;
			//m_nbOutputSlices = 2 + (m_freqMax - m_freqMin) / m_freqStep;
		}
	}

	void setFreqMin(int freqMin) {
		if ( m_freqMin == freqMin )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_freqMin = freqMin;
			m_isComputed = false;
			//m_nbOutputSlices = 2 + (m_freqMax - m_freqMin) / m_freqStep;
		}
	}

	void setFreqStep(int freqStep = 2) {
		if ( m_freqStep == freqStep )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_freqStep = freqStep;
			m_isComputed = false;
			//m_nbOutputSlices = 2 + (m_freqMax - m_freqMin) / m_freqStep;
		}
	}

	void setDTauReference(long dtau) {
		if ( m_dtauReference == dtau )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_dtauReference = dtau;
			m_isComputed = false;
			//m_nbOutputSlices = 2 + (m_freqMax - m_freqMin) / m_freqStep;
		}
	}

protected:

	//int m_geologicalTime = 0;
	std::vector<RgtSeed> m_seeds;
	int m_distancePower = 8;
	bool m_polarity = true;
	bool m_useSnap = false;
	int m_snapWindow = 3;
	bool m_useMedian = false;
	int m_lwx_medianFilter = 11;

	int m_freqMin = 30;
	int m_freqMax = 150;
	int m_freqStep = 2;
	long m_dtauReference = 0;
	std::vector<float> m_constrainIso;
	std::vector<ReferenceDuo> m_reference;
};

template<typename InputCubeType>
class MorletProcess : public ToAnalyse3Process {
public:
	MorletProcess(Seismic3DDataset *cubeS, int channelS, Seismic3DDataset *cubeT, int channelT);
	virtual ~MorletProcess();

	void init() override;

	bool isCompatible(e_SliceDirection lastDirection, int lastSlice);
//	std::pair<int, int> computeRange(int compo);
//	void computeHistogram(
//			int rangeMin, int rangeMax, int nBuckets, int compo, double* histo );
//	void setComponentRange( int component, int rangeMin, int rangeMax );
	void saveIn(Seismic3DAbstractDataset * modulesDataset);


private:
	void cefft ( std::complex<float>* x, int* ane, int* asign);
};

#include "MorletProcess.hpp"
#endif /* MURATPROCESSLIB_SRC_MorletProcess_H_ */

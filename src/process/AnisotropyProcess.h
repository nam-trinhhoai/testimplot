/*

 *
 *  Created on: 29 mars 2020
 *      Author: Georges
 */

#ifndef SRC_PROCESS_AnisotropyProcess_H_
#define SRC_PROCESS_AnisotropyProcess_H_

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
#include "seismic3ddataset.h"
#include "process/RgtLayerProcessUtil.h"

class AnisotropyAbstractProcess : public LayerProcess {

public:
	AnisotropyAbstractProcess(
			Seismic3DDataset *cubeS, int channelS,
					Seismic3DDataset *cubeT, int channelT);
	virtual ~AnisotropyAbstractProcess() {};
	virtual void init() {};

	// useless now but may be used in the future
	int getWindowSize() const {
		return m_windozSize;
	}

	void setWindowSize(int windozSize) {
		if ( m_windozSize == windozSize )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_windozSize = windozSize;
			//m_nbOutputSlices = 2 + windozSize;
			m_isComputed = false;
		}
	}

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
		if ( m_distancePower == dist )
			return;

		{
			QMutexLocker lock(&m_computeMutex);
			m_distancePower = dist;
			m_isComputed = false;
		}
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
		if (m_snapWindow==val) {
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
		if (m_lwx_medianFilter == lwx) {
			return;
		}
		{
			QMutexLocker lock(&m_computeMutex);
			m_lwx_medianFilter = lwx;
			m_isComputed = false;
		}
	}

	void setDTauReference(long dtau) {
		if (m_dtauReference == dtau) {
			return;
		}
		{
			QMutexLocker lock(&m_computeMutex);
			m_dtauReference = dtau;
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

    void setAttributDatasets(const QList<std::tuple<Seismic3DDataset*, int, float>>& datasets) {
        QMutexLocker lock(&m_computeMutex);
    	m_attributDatasets = datasets;
        m_isComputed = false;
    }

	int getNbOutputSlices() const {
		QMutexLocker lock(&m_cacheMutex);
		return m_nbOutputSlices;
	}

	void compute(LayerSpectrumDialog *layerspectrumdialog) override;

	/**
	 * Give a pointer to the cached result for the specified spectrumSlice
	 *
	 * ! 0 is used for rgt index
	 * ! 1 is used for cubeS values
	 * ! 2 -> 1 + window/2 is the process result
	 *
	 * The pointer given will be invalid if spectrumSlice > getNbOutputSlices()
	 */
	const float* getModuleData(std::size_t spectrumSlice) const;

protected:

	//int m_geologicalTime = 0;
	std::vector<RgtSeed> m_seeds;
	int m_distancePower = 8;
	bool m_polarity = true;
	bool m_useSnap = false;
	int m_snapWindow = 3;
	bool m_useMedian = false;
	int m_lwx_medianFilter = 11;
	int m_windozSize = 64;

	long m_dtauReference = 0;
	std::vector<ReferenceDuo> m_reference;
	std::vector<float> m_constrainIso;

	QList<std::tuple<Seismic3DDataset*, int, float>> m_attributDatasets;
	private:
		template <typename RgtType>
		struct ComputeModulesKernel {
			static std::vector<std::vector<float>>* run(AnisotropyAbstractProcess* pr,
					const std::vector<float>& constrainLayer,
					std::vector<ReferenceDuo>& reference,
					const std::vector<RgtSeed>& seeds,
					long dtauReference,
					bool useSnap, bool useMedian, int distancePower,
					const QList<std::tuple<Seismic3DDataset*, int, float>>& attributsDatasets,
					int windowSize, LayerSpectrumDialog *layerspectrumdialog);

			template <typename SeismicType>
			struct ComputeModulesKernelLevel2 {
				static std::vector<std::vector<float>>* run(AnisotropyAbstractProcess* pr,
						const std::vector<float>& constrainLayer,
						std::vector<ReferenceDuo>& reference,
						std::vector<RgtSeed> seeds,
						long dtauReference,
						bool useSnap, bool useMedian, int distancePower,
						const QList<std::tuple<Seismic3DDataset*, int, float>>& attributsDatasets,
						int windowSize, LayerSpectrumDialog *layerspectrumdialog);
			};
		};
};

template<typename InputCubeType>
class AnisotropyProcess : public AnisotropyAbstractProcess {
public:
	AnisotropyProcess(
		Seismic3DDataset *cubeS, int channelS,
		Seismic3DDataset *cubeT, int channelT);
	virtual ~AnisotropyProcess();

	void init() override;

	bool isCompatible(e_SliceDirection lastDirection, int lastSlice);
//	std::pair<int, int> computeRange(int compo);
//	void computeHistogram(
//			int rangeMin, int rangeMax, int nBuckets, int compo, double* histo );
//	void setComponentRange( int component, int rangeMin, int rangeMax );
	void saveIn(Seismic3DAbstractDataset* modulesDataset);


private:
	void cefft ( std::complex<float>* x, int* ane, int* asign);
};

#include "AnisotropyProcess.hpp"
#endif /* SRC_PROCESS_AnisotropyProcess_H_ */

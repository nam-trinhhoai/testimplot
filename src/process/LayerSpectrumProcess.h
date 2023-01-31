/*

 *
 *  Created on: 5 janv. 2018
 *      Author: Georges
 */

#ifndef MURATPROCESSLIB_SRC_LayerSpectrumProcess_H_
#define MURATPROCESSLIB_SRC_LayerSpectrumProcess_H_

#include "LayerProcess.h"

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
#include "slicer/data/sliceutils.h"

class Seismic3DDataset;
class Seismic3DAbstractDataset;

class ToAnalyse2Process : public LayerProcess {

public:
	ToAnalyse2Process(Seismic3DDataset* cubeS, int channelS, Seismic3DDataset* cubeT, int channelT);
	virtual ~ToAnalyse2Process() {};
	virtual void init() {};

	int getWindowSize() const {
		return m_windozSize;
	}

	void setWindowSize(int windozSize) {
		if ( m_windozSize == windozSize )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_windozSize = windozSize;
			//m_nbOutputSlices = 2 + windozSize /2;
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

	void setHatPower(float val) {
		if (m_hatPower == val) {
			return;
		} else {
			QMutexLocker lock(&m_computeMutex);
			m_hatPower = val;
			m_isComputed = false;
		}
	}

	void setDTauReference(long val) {
		if (m_dtauReference == val) {
			return;
		} else {
			QMutexLocker lock(&m_computeMutex);
			m_dtauReference = val;
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
	const float* getModuleData(std::size_t spectrumSlice) const override;



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
	float m_hatPower = 5;
	long m_dtauReference = 0;
	std::vector<float> m_constrainIso;
	std::vector<ReferenceDuo> m_reference;

private:
	std::vector<std::vector<float>>* computeModules( ToAnalyse2Process* pr, int windowSize,  std::vector<RgtSeed> seeds, float hat_pow, bool polarity,
				bool useSnap, bool useMedian, int lwx, int distancePower,
				int snapWindow,const std::vector<float>& constrainLayer, long dtauReference,
				std::vector<ReferenceDuo>& reference, LayerSpectrumDialog *layerspectrumdialog);
	std::vector<std::vector<float>>* computeModulesCwt( ToAnalyse2Process* pr, int windowSize,  std::vector<RgtSeed> seeds, float hat_pow, bool polarity,
				bool useSnap, bool useMedian, int lwx, int distancePower,
				int snapWindow,const std::vector<float>& constrainLayer, long dtauReference, std::string cwtSeismicPath,
				std::string cwtRgtPath, std::vector<ReferenceDuo>& reference,
				LayerSpectrumDialog *layerspectrumdialog);
	template <typename RgtType>
	struct ComputeModulesDefaultKernel {
		static std::vector<std::vector<float>>* run( ToAnalyse2Process* pr, int windowSize, const std::vector<RgtSeed>& seeds, float hat_pow, bool polarity,
				bool useSnap, bool useMedian, int lwx, int distancePower,
				int snapWindow,const std::vector<float>& constrainLayer, long dtauReference, std::vector<ReferenceDuo>& reference,
				LayerSpectrumDialog *layerspectrumdialog);

		template <typename SeismicType>
		struct ComputeModulesDefaultKernelLevel2 {
			static std::vector<std::vector<float>>* run( ToAnalyse2Process* pr, int windowSize,  std::vector<RgtSeed> seeds, float hat_pow, bool polarity,
					bool useSnap, bool useMedian, int lwx, int distancePower,
					int snapWindow,const std::vector<float>& constrainLayer, long dtauReference, std::vector<ReferenceDuo>& reference,
					LayerSpectrumDialog *layerspectrumdialog);
		};
	};

};

template<typename InputCubeType>
class LayerSpectrumProcess : public ToAnalyse2Process {
public:
	LayerSpectrumProcess(Seismic3DDataset *cubeS, int channelS, Seismic3DDataset *cubeT, int channelT);
	virtual ~LayerSpectrumProcess();

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

#include "LayerSpectrumProcess.hpp"
#endif /* MURATPROCESSLIB_SRC_LayerSpectrumProcess_H_ */

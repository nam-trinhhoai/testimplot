/*

 *
 *  Created on: 24 janv. 2021
 *      Author: Armand
 */

#ifndef SRC_PROCESS_KOHONENLAYERPROCESS_H_
#define SRC_PROCESS_KOHONENLAYERPROCESS_H_

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
#include "iabstractisochrone.h"

class Seismic3DDataset;
class Seismic3DAbstractDataset;

class KohonenLayerProcess : public LayerProcess {

public:
	KohonenLayerProcess(Seismic3DDataset* cubeS, int channelS, Seismic3DDataset* cubeT, int channelT);
	virtual ~KohonenLayerProcess() {};
	virtual void init() {};

	void setTmapExampleSize(int val) {
		if ( m_tmapExampleSize == val )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_tmapExampleSize = val;
			m_isComputed = false;
		}
	}

	void setTmapSize(int val) {
		if ( m_tmapSize == val )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_tmapSize = val;
			m_isComputed = false;
		}
	}

	void setTmapExampleStep(int val) {
		if ( m_tmapExampleStep == val )
			return;
		{
			QMutexLocker lock(&m_computeMutex);
			m_tmapExampleStep = val;
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
	 * ! 2 is used for the labels
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
	long m_dtauReference = 0;
	int m_tmapExampleSize = 10;
	int m_tmapSize = 33;
	int m_tmapExampleStep = 20;
	std::vector<float> m_constrainIso;
	std::vector<ReferenceDuo> m_reference;

private:
	std::vector<std::vector<float>>* computeModules(std::vector<RgtSeed> seeds, bool polarity,
				bool useSnap, bool useMedian, int lwx, int distancePower,
				int snapWindow,const std::vector<float>& constrainLayer, long dtauReference,
				std::vector<ReferenceDuo>& reference, LayerSpectrumDialog *layerspectrumdialog);

};

class SeedLayerIsochrone : public IAbstractIsochrone {
public:
	SeedLayerIsochrone(Seismic3DDataset* cubeT, int channel, const std::vector<RgtSeed>& seeds, bool polarity,
			bool useSnap, bool useMedian, int lwx, int distancePower,
			int snapWindow,const std::vector<float>& constrainLayer, long dtauReference,
			std::vector<ReferenceDuo>& reference);
	virtual ~SeedLayerIsochrone();

	virtual int getNumTraces() const override;
	virtual int getNumProfils() const override;

	virtual float getValue(long i, long j, bool* ok) override;
	virtual float* getTab() override;

private:
	template<typename InputCubeType>
	struct GetIsochroneInlineKernel {
		static void run(SeedLayerIsochrone* obj, long j, float* tab);
	};

	long m_numTraces;
	long m_numProfils;
	std::vector<float> m_fullMapBuffer;
	long m_N;

	Seismic3DDataset* m_cubeT;
	int m_channel;
	std::vector<RgtSeed> m_seeds;
	bool m_polarity;
	bool m_useSnap;
	bool m_useMedian;
	int m_lwx;
	int m_distancePower;
	int m_snapWindow;
	const std::vector<float>& m_constrainLayer;
	long m_dtauReference;
	std::vector<ReferenceDuo>& m_reference;
	std::vector<bool> m_readInlines;
};

#endif

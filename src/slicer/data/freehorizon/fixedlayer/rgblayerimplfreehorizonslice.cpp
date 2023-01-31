#include "rgblayerimplfreehorizonslice.h"
#include "LayerSlice.h"
#include "rgblayerslicegraphicrepfactory.h"
#include "cudargbimage.h"
#include "cudaimagepaletteholder.h"
#include "cpuimagepaletteholder.h"
#include "seismic3ddataset.h"
#include "affine2dtransformation.h"
#include <rgtSpectrumGetBestComponent.h>
#include "rgblayerfreehorizongraphicrepfactory.h"
#include "rgblayerimplfreehorizonslice.h"
#include "rgblayerfreehorizongraphicrepfactory.h"




#include <cstring>
#include <cmath>
#include <QCoreApplication>

RGBLayerImplFreeHorizonOnSlice::RGBLayerImplFreeHorizonOnSlice(WorkingSetManager *workingSet,
		LayerSlice *layerSlice, QObject *parent) :
		IData(workingSet, parent), m_layerSlice(layerSlice) {
	m_image.reset(new CUDARGBImage(layerSlice->width(), layerSlice->depth(),
			ImageFormats::QSampleType::FLOAT32, layerSlice->seismic()->ijToXYTransfo(),
			this));

	m_repFactory.reset(new RGBLayerFreeHorizonGraphicRepFactory(this));
	resetFrequencies();

	connect(m_layerSlice, &LayerSlice::computationFinished, this,
		&RGBLayerImplFreeHorizonOnSlice::updateFromComputation);
	connect(m_layerSlice, &LayerSlice::deleteRgtLayer, this,
			&RGBLayerImplFreeHorizonOnSlice::deleteRep);
}

RGBLayerImplFreeHorizonOnSlice::~RGBLayerImplFreeHorizonOnSlice() {

}


int RGBLayerImplFreeHorizonOnSlice::redIndex() const {
	return m_freqIndex[0];
}

int RGBLayerImplFreeHorizonOnSlice::greenIndex() const {
	return m_freqIndex[1];
}

int RGBLayerImplFreeHorizonOnSlice::blueIndex() const {
	return m_freqIndex[2];
}

void RGBLayerImplFreeHorizonOnSlice::setRedIndex(int value) {
	m_freqIndex[0] = value;
	loadSlice();
	if (m_locked) {
		m_lockedRedIndex = m_freqIndex[0];
	}
	emit frequencyChanged();
}

void RGBLayerImplFreeHorizonOnSlice::setGreenIndex(int value) {
	m_freqIndex[1] = value;
	loadSlice();
	if (m_locked) {
		m_lockedGreenIndex = m_freqIndex[1];
	}
	emit frequencyChanged();
}

void RGBLayerImplFreeHorizonOnSlice::setBlueIndex(int value) {
	m_freqIndex[2] = value;
	loadSlice();
	if (m_locked) {
		m_lockedBlueIndex = m_freqIndex[2];
	}
	emit frequencyChanged();
}

void RGBLayerImplFreeHorizonOnSlice::setRGBIndexes(int red, int green, int blue) {
	m_freqIndex[0] = red;
	m_freqIndex[1] = green;
	m_freqIndex[2] = blue;
	loadSlice();
	if (m_locked) {
		m_lockedRedIndex = m_freqIndex[0];
		m_lockedGreenIndex = m_freqIndex[1];
		m_lockedBlueIndex = m_freqIndex[2];
	}
	emit frequencyChanged();
}

QUuid RGBLayerImplFreeHorizonOnSlice::dataID() const {
return m_layerSlice->dataID();
}

QString RGBLayerImplFreeHorizonOnSlice::name() const {
	return "RGB " + m_layerSlice->name();
}

//IData
IGraphicRepFactory* RGBLayerImplFreeHorizonOnSlice::graphicRepFactory() {
	return m_repFactory.get();
}

void RGBLayerImplFreeHorizonOnSlice::loadSlice() {
	const float* tabR = nullptr, *tabG = nullptr, *tabB = nullptr;
	for (int e=0; e<3; e++) {
		if (m_freqIndex[e]>=m_layerSlice->getNbOutputSlices()) {
			m_freqIndex[e] = m_layerSlice->getNbOutputSlices() - 1;
		}
	}
	tabR = m_layerSlice->getModuleData(m_freqIndex[0]);
	tabG = m_layerSlice->getModuleData(m_freqIndex[1]);
	tabB = m_layerSlice->getModuleData(m_freqIndex[2]);

	if (!tabR || !tabG || !tabB)
		return;

	QCoreApplication::processEvents();
	m_image->get(0)->updateTexture(tabR, false);
	m_image->get(1)->updateTexture(tabG, false);
	m_image->get(2)->updateTexture(tabB, false);

	std::size_t N = m_image->width()*m_image->height();

	/*
	m_image->get(0)->lockPointer();
	float* tab = static_cast<float*>(m_image->get(0)->backingPointer());
	float minVal = 32000;
	float maxVal = -32000;
	for(std::size_t index=0; index<N; index++) {
		if (minVal>tab[index]) {
			minVal = tab[index];
		}
		if (maxVal<tab[index]) {
			maxVal = tab[index];
		}
	}
	//qDebug() << "Load Slice red min max :"  << minVal << maxVal;
	m_image->get(0)->unlockPointer();
	*/
//	fprintf(stderr, "%s %d\n", __FILE__, __LINE__);

	if (m_locked) {
		m_image->setRange(0, m_lockedRedRange);
		m_image->setRange(1, m_lockedGreenRange);
		m_image->setRange(2, m_lockedBlueRange);
	}
}

void RGBLayerImplFreeHorizonOnSlice::resetFrequencies() {
	const float* tab = m_layerSlice->getModuleData(0);
	if (tab==nullptr) {
		m_freqIndex[0] = 2;
		m_freqIndex[1] = 3;
		m_freqIndex[2] = 4;
	} else {
        int greenLayer = getMaximumEnergyComponent();
        if (greenLayer==2) {
        	greenLayer = 4;
        }
        int redLayer = greenLayer - 2;
        int blueLayer = greenLayer + 2;
        if (redLayer<2) {
        	redLayer = 2;
        }

        if (greenLayer>=m_layerSlice->getNbOutputSlices()-1) {
        	greenLayer = m_layerSlice->getNbOutputSlices() - 2;
        }
        if (greenLayer<2) {
        	greenLayer = 2;
        }

        if (blueLayer>=m_layerSlice->getNbOutputSlices()) {
        	blueLayer = m_layerSlice->getNbOutputSlices() - 1;
        }
		m_freqIndex[0] = redLayer;
		m_freqIndex[1] = greenLayer;
		m_freqIndex[2] = blueLayer;
	}
	loadSlice();
	if (m_locked) {
		m_lockedRedIndex = m_freqIndex[0];
		m_lockedGreenIndex = m_freqIndex[1];
		m_lockedBlueIndex = m_freqIndex[2];
	}
	emit frequencyChanged();
}



// original
int RGBLayerImplFreeHorizonOnSlice::getMaximumEnergyComponent() {
	int bestComponent = 0;
	int bestRange = 0;
	std::size_t N = m_image->width() * m_image->height();

	int n1 = m_image->width();
	int n2 = m_image->height();

	std::vector<int> range;
	range.resize(m_layerSlice->getNbOutputSlices()-2);

	#pragma omp parallel for
	for (std::size_t component=2; component<m_layerSlice->getNbOutputSlices(); component++) {
		const float* tab = m_layerSlice->getModuleData(component);
		float min = std::numeric_limits<short>::max();
		float max = std::numeric_limits<short>::min();

		std::vector<long> histogram;
		std::size_t N_histo = 65536;

		histogram.resize(N_histo);
		//double step = (static_cast<double>(std::numeric_limits<float>::max()) - std::numeric_limits<float>::min()) / N;
		double step = 1;
		double dmin = static_cast<double>(std::numeric_limits<short>::min());
		std::memset(histogram.data(), 0, N_histo*sizeof(long) );

		for (std::size_t i=0; i<N; i++) {
			if (max<tab[i]) {
				max = tab[i];
			}
			if (min>tab[i]) {
				min = tab[i];
			}
			int index = static_cast<int>(std::floor((tab[i] - dmin) / step));
			if (index>=((long)N_histo)) {
				index = N_histo-1;
			} else if (index<0) {
				index = 0;
			}
			histogram[index] ++;
		}

		double vmin, vmax;
		if (min!=max) {
			double borneInf = 0.005;

			int i=0;
			std::size_t cumul = histogram[0];
			while(cumul==0 || cumul<borneInf*N) {
				i++;
				if (i<N_histo) {
					cumul += histogram[i];
				}
			}
			vmin = std::numeric_limits<float>::min() + i * step;

			i = N_histo-1;
			std::size_t sum = N;
			cumul = histogram[N_histo-1];
			while(cumul==0 || cumul<borneInf*sum) {
				i--;
				if (i>0) {
					cumul += histogram[i];
				}
			}
			vmax = std::numeric_limits<float>::min() + i * step;
			if (vmax<vmin) {
				vmax = vmin;
			}
		} else {
			vmin = min;
			vmax = max;
		}
//		if (vmax-vmin>bestRange) {
//			bestRange = vmax - vmin;
//			bestComponent = component;
//		}
		range[component-2] = vmax - vmin;
	}



	for (std::size_t component=2; component<m_layerSlice->getNbOutputSlices(); component++) {
		if (range[component-2]>bestRange) {
			bestRange = range[component-2];
			bestComponent = component;
		}

		//fprintf(stderr, "range: %d\n", range[component-2]);

	}
	return bestComponent;
}



/*
int RGBLayerImplFreeHorizonOnSlice::getMaximumEnergyComponent() {
	int bestComponent = 0;
	int bestRange = 0;
	std::size_t N = m_image->width() * m_image->height();

	int n1 = m_image->width();
	int n2 = m_image->height();

	std::vector<int> range;
	range.resize(m_layerSlice->getNbOutputSlices()-2);

	#pragma omp parallel for
	for (std::size_t component=2; component<m_layerSlice->getNbOutputSlices(); component++) {
		const float* tab = m_layerSlice->getModuleData(component);
		range[component-2] = rgtSpectrumGetMaximumEnergyComponent((float*)tab, N);
	}

	for (std::size_t component=2; component<m_layerSlice->getNbOutputSlices(); component++) {
			if (range[component-2]>bestRange) {
				bestRange = range[component-2];
				bestComponent = component;
			}


	fprintf(stderr, "range: %d\n", range[component]);
	}
	return bestComponent;
}
*/



/*

int RGBLayerImplFreeHorizonOnSlice::getMaximumEnergyComponent(float *tab, size_t size) {

	size_t N = size;
	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::min();

	std::vector<long> histogram;
	std::size_t N_histo = 65536;

	histogram.resize(N_histo);
		//double step = (static_cast<double>(std::numeric_limits<float>::max()) - std::numeric_limits<float>::min()) / N;
	double step = 1;
	double dmin = static_cast<double>(std::numeric_limits<float>::min());
	std::memset(histogram.data(), 0, N_histo*sizeof(long) );

	for (std::size_t i=0; i<N; i++) {
		if (max<tab[i]) {
			max = tab[i];
		}
		if (min>tab[i]) {
			min = tab[i];
		}

		int index = static_cast<int>(std::floor((tab[i] - dmin) / step));
		if (index>=N_histo) {
			index = N_histo-1;
		} else if (index<0) {
			index = 0;
		}
		histogram[index] ++;
	}

	double vmin, vmax;
	if (min!=max) {
		double borneInf = 0.005;

		int i=0;
		std::size_t cumul = histogram[0];
		while(cumul==0 || cumul<borneInf*N) {
			i++;
			if (i<N_histo) {
				cumul += histogram[i];
			}
		}

		vmin = std::numeric_limits<float>::min() + i * step;
		i = N_histo-1;
		std::size_t sum = N;
		cumul = histogram[N_histo-1];
		while(cumul==0 || cumul<borneInf*sum) {
			i--;
			if (i>0) {
				cumul += histogram[i];
			}
		}

		vmax = std::numeric_limits<float>::min() + i * step;
		if (vmax<vmin) {
			vmax = vmin;
		}
	} else {
		vmin = min;
		vmax = max;
	}

	return vmax - vmin;
}
*/



void RGBLayerImplFreeHorizonOnSlice::deleteRep() {
    emit deletedRep();
}

bool RGBLayerImplFreeHorizonOnSlice::isLocked() const {
	return m_locked;
}

int RGBLayerImplFreeHorizonOnSlice::lockedRedIndex() const {
	return m_lockedRedIndex;
}

int RGBLayerImplFreeHorizonOnSlice::lockedGreenIndex() const {
	return m_lockedGreenIndex;
}

int RGBLayerImplFreeHorizonOnSlice::lockedBlueIndex() const {
	return m_lockedBlueIndex;
}

QVector2D RGBLayerImplFreeHorizonOnSlice::lockedRedRange() const {
	return m_lockedRedRange;
}

QVector2D RGBLayerImplFreeHorizonOnSlice::lockedGreenRange() const {
	return m_lockedGreenRange;
}

QVector2D RGBLayerImplFreeHorizonOnSlice::lockedBlueRange() const {
	return m_lockedBlueRange;
}

void RGBLayerImplFreeHorizonOnSlice::setLockedRedRange(const QVector2D& redRange) {
	m_lockedRedRange = redRange;
	if (m_locked) {
		m_image->setRange(0, m_lockedRedRange);
	}
}

void RGBLayerImplFreeHorizonOnSlice::setLockedGreenRange(const QVector2D& greenRange) {
	m_lockedGreenRange = greenRange;
	if (m_locked) {
		m_image->setRange(1, m_lockedGreenRange);
	}
}

void RGBLayerImplFreeHorizonOnSlice::setLockedBlueRange(const QVector2D& blueRange) {
	m_lockedBlueRange = blueRange;
	if (m_locked) {
		m_image->setRange(2, m_lockedBlueRange);
	}
}

void RGBLayerImplFreeHorizonOnSlice::unlock() {
	if (m_locked) {
		m_locked = false;
		emit lockChanged();
	}
}

void RGBLayerImplFreeHorizonOnSlice::lock() {
	if (!m_locked) {
		m_lockedRedIndex = redIndex();
		m_lockedRedRange = m_image->get(0)->range();
		m_lockedGreenIndex = greenIndex();
		m_lockedGreenRange = m_image->get(1)->range();
		m_lockedBlueIndex = blueIndex();
		m_lockedBlueRange = m_image->get(2)->range();

		m_locked = true;
		emit lockChanged();
	}
}

void RGBLayerImplFreeHorizonOnSlice::updateFromComputation() {
	if (m_locked) {
		loadSlice();
	} else {
		resetFrequencies();
	}
}

bool RGBLayerImplFreeHorizonOnSlice::isMinimumValueActive() const {
	return m_useMinimumValue;
}

void RGBLayerImplFreeHorizonOnSlice::setMinimumValueActive(bool active, bool bypassLock) {
	if (m_useMinimumValue!=active && (!m_locked || bypassLock)) {
		m_useMinimumValue = active;
		emit minimumValueActivated(m_useMinimumValue);
	}
}

float RGBLayerImplFreeHorizonOnSlice::minimumValue() const {
	return m_minimumValue;
}

void RGBLayerImplFreeHorizonOnSlice::setMinimumValue(float minimumValue, bool bypassLock) {
	if (m_minimumValue!=minimumValue && (!m_locked || bypassLock)) {
		m_minimumValue = minimumValue;
		emit minimumValueChanged(m_minimumValue);
	}
}

IsoSurfaceBuffer RGBLayerImplFreeHorizonOnSlice::getIsoBuffer()
{
	return m_layerSlice->getIsoBuffer();
}


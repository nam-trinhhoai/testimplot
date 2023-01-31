/*
 * LayerSlice.cpp
 *
 *  Created on: Jun 15, 2020
 *      Author: l0222891
 */

#include "LayerSlice.h"

#include "seismic3ddataset.h"
#include "layerslicegrahicrepfactory.h"
#include "cudaimagepaletteholder.h"
#include "cpuimagepaletteholder.h"
#include "cuda_volume.h"
#include "seismic3dcudadataset.h"
#include "seismic3ddataset.h"
#include  "cudargttile.h"
#include <iostream>
#include <QElapsedTimer>
#include <QCoreApplication>
#include "cuda_common_helpers.h"
#include "datasetbloccache.h"
#include "affine2dtransformation.h"

#include "affinetransformation.h"
#include "fixedlayerfromdataset.h"

#include "LayerSpectrumDialog.h"
#include "abstractgraphicrep.h"
#include "textcolortreewidgetitemdecorator.h"
#include "workingsetmanager.h"

#define TILE_SIZE 128
#define EXTRACTION_INIT_WINDOW 21

LayerSlice::LayerSlice(WorkingSetManager *workingSet,
		Seismic3DDataset *datasetS, int channelS,
		Seismic3DDataset *datasetT, int channelT,
		int iDataType,
		QObject *parent):
		IData(workingSet, parent) {
	m_datasetS = datasetS;
	m_channelS = channelS;
	m_datasetT = datasetT;
	m_channelT = channelT;
	m_simplifyMeshSteps = 10;
	m_compressionMesh =0;
	m_repFactory.reset(new LayerSliceGraphicRepFactory(this));
//	m_currentFile = nullptr;
//	m_headerLength = 0;

	m_extractionWindow = EXTRACTION_INIT_WINDOW;

	m_isoSurfaceHolder.reset(new CUDAImagePaletteHolder(width(), depth(),
			ImageFormats::QSampleType::FLOAT32, m_datasetS->ijToXYTransfo(),
			parent));
	m_image.reset(new CUDAImagePaletteHolder(width(), depth(),
			ImageFormats::QSampleType::FLOAT32, m_datasetS->ijToXYTransfo(),
			parent));

	//TODO WARNING pas propre: PROCESS instances should be created according to method used...
	// Ceci est de la dauble, mais il y a le feu!
	m_layerSpectrumProcess.reset(new LayerSpectrumProcess<short>(datasetS, channelS, datasetT, channelT));
	m_morletProcess.reset(new MorletProcess<short>(datasetS, channelS, datasetT, channelT));
	m_gccProcess.reset(new GradientMultiScaleProcess<short>(datasetS, channelS, datasetT, channelT));
	m_tmapProcess.reset(new KohonenLayerProcess(datasetS, channelS, datasetT, channelT));
	m_attributProcess.reset(new AttributProcess<short>(datasetS, channelS, datasetT, channelT));
	m_anisotropyProcess.reset(new AnisotropyProcess<short>(datasetS, channelS, datasetT, channelT));

	QVector2D minMax = rgtMinMax();

	// MZR 09072021
	switch(iDataType)
	{
	case eComputeMethd_Morlet:
		m_Name = "Morlet RGT " + m_datasetS->name();
		m_Comptype = iDataType;
		break;
	case eComputeMethd_Spectrum :
		m_Name = "Spectrum RGT " + m_datasetS->name();
		m_Comptype = iDataType;
		break;
	case eComputeMethd_Gcc :
		m_Name = "GCC RGT " + m_datasetS->name();
		m_Comptype = iDataType;
		break;
	case eComputeMethd_TMAP :
		m_Name = "TMAP RGT " + m_datasetS->name();
		m_Comptype = iDataType;
		break;
	case eComputeMethd_Mean :
		m_Name = "MEAN RGT " + m_datasetS->name();
		m_Comptype = iDataType;
		break;
	case eComputeMethd_Anisotropy :
		m_Name = "Anisotropy RGT " + m_datasetS->name();
		m_Comptype = iDataType;
		break;
	default:
		m_Name = "RGT " + m_datasetS->name();
		m_Comptype = iDataType;
		break;
	}
	loadSlice(minMax[0]);

	m_conn.push_back(connect(m_layerSpectrumProcess.get(), &LayerProcess::processCacheIsReset, [this]() {
		std::size_t N = getNbOutputSlices();
		if (m_currentSlice>=N ) {
			m_currentSlice = N - 1;
		}
		loadSlice(m_isoSurfaceHolder.get(), m_image.get(), m_extractionWindow, m_currentSlice);
		QCoreApplication::processEvents();
		emit computationFinished(N);
	}));

	// JD TODO
	m_conn.push_back(connect(m_gccProcess.get(), &LayerProcess::processCacheIsReset, [this]() {
		std::size_t N = getNbOutputSlices();
		if (m_currentSlice>=N ) {
			m_currentSlice = N - 1;
		}
		loadSlice(m_isoSurfaceHolder.get(), m_image.get(), m_extractionWindow, m_currentSlice);
		QCoreApplication::processEvents();
		emit computationFinished(N);
	}));

	m_conn.push_back(connect(m_tmapProcess.get(), &LayerProcess::processCacheIsReset, [this]() {
		std::size_t N = getNbOutputSlices();
		if (m_currentSlice>=N ) {
			m_currentSlice = N - 1;
		}
		loadSlice(m_isoSurfaceHolder.get(), m_image.get(), m_extractionWindow, m_currentSlice);
		QCoreApplication::processEvents();
		emit computationFinished(N);
	}));

	m_conn.push_back(connect(m_attributProcess.get(), &LayerProcess::processCacheIsReset, [this]() {
		std::size_t N = getNbOutputSlices();
		if (m_currentSlice>=N ) {
			m_currentSlice = N - 1;
		}
		loadSlice(m_isoSurfaceHolder.get(), m_image.get(), m_extractionWindow, m_currentSlice);
		QCoreApplication::processEvents();
		emit computationFinished(N);
	}));

	m_conn.push_back(connect(m_anisotropyProcess.get(), &LayerProcess::processCacheIsReset, [this]() {
		std::size_t N = getNbOutputSlices();
		if (m_currentSlice>=N ) {
			m_currentSlice = N - 1;
		}
		loadSlice(m_isoSurfaceHolder.get(), m_image.get(), m_extractionWindow, m_currentSlice);
		QCoreApplication::processEvents();
		emit computationFinished(N);
	}));

	m_decorator = nullptr;
}

int LayerSlice::getComputationType() const {
    return m_Comptype;
}

void LayerSlice::deleteRgt(){
	emit deleteRgtLayer();
}

uint LayerSlice::extractionWindow() const {
	return m_extractionWindow;
}

void LayerSlice::setExtractionWindow(uint w) {
	m_extractionWindow = w;
	loadSlice(m_currentSlice);
	emit extractionWindowChanged(w);
}

int LayerSlice::currentPosition() const {
	return m_currentSlice;
}
void LayerSlice::setSlicePosition(int pos) {
	loadSlice(pos);
	emit RGTIsoValueChanged(pos);
}

QString LayerSlice::getCurrentLabel() const {
	return getLabelFromPosition(m_currentSlice);
}

void LayerSlice::loadSlice(unsigned int z) {
	m_currentSlice = z;
	loadSlice(m_image.get(), m_extractionWindow, z);

	QString label = getLabelFromPosition(m_currentSlice);
	std::map<QString, PaletteParameters>::const_iterator it = std::find_if(m_cachedPaletteParameters.begin(),
			m_cachedPaletteParameters.end(), [label](const std::pair<QString, PaletteParameters>& pair) {
		return pair.first.compare(label)==0;
	});
	if (it!=m_cachedPaletteParameters.end()) {
		m_image->setRange(it->second.range);
		m_image->setLookupTable(it->second.lookupTable);
	}
}


// JD TODO
void LayerSlice::loadSlice(CUDAImagePaletteHolder *isoSurfaceImage,
		CUDAImagePaletteHolder *image, unsigned int extractionWindow,
		unsigned int z) {
	//TODO
	const float* tab = nullptr, *tab0 = nullptr;
	if ( m_layerSpectrumProcess && m_method==1 ) {
		tab = m_layerSpectrumProcess->getModuleData(z);
		tab0 = m_layerSpectrumProcess->getModuleData(0);
	} else if ( m_gccProcess && m_method==2 )
	{
		tab = m_gccProcess->getModuleData(z);
		tab0 = m_gccProcess->getModuleData(0);
	} else if ( m_tmapProcess && m_method==3 )
	{
		tab = m_tmapProcess->getModuleData(z);
		tab0 = m_tmapProcess->getModuleData(0);
	} else if ( m_attributProcess && m_method==4 )
	{
		tab = m_attributProcess->getModuleData(z);
		tab0 = m_attributProcess->getModuleData(0);
	} else if ( m_anisotropyProcess && m_method==5 )
	{
		tab = m_anisotropyProcess->getModuleData(z);
		tab0 = m_anisotropyProcess->getModuleData(0);
	}

	if (!tab || !tab0)
		return;

//	int d = m_datasetS->depth();
//	int w = m_datasetS->width();
//	int h = m_datasetS->height();
//
//	Seismic3DDataset *seismic = dynamic_cast<Seismic3DDataset*>(m_datasetS);
//	Seismic3DDataset *rgt = dynamic_cast<Seismic3DDataset*>(m_datasetT);
	QCoreApplication::processEvents();

	isoSurfaceImage->updateTexture(tab0, false);
	image->updateTexture(tab, false);
}

void LayerSlice::loadSlice( CUDAImagePaletteHolder *image,
		unsigned int extractionWindow, unsigned int z) {
	const float* tab = nullptr;
	if ( m_layerSpectrumProcess && m_method==1 ) {
		tab = m_layerSpectrumProcess->getModuleData(z);
	} else if ( m_gccProcess && m_method==2 )
	{
		tab = m_gccProcess->getModuleData(z);
	} else if ( m_tmapProcess && m_method==3 )
	{
		tab = m_tmapProcess->getModuleData(z);
	} else if ( m_attributProcess && m_method==4 )
	{
		tab = m_attributProcess->getModuleData(z);
	} else if ( m_anisotropyProcess && m_method==5 )
	{
		tab = m_anisotropyProcess->getModuleData(z);
	}

	if (!tab)
		return;
	QCoreApplication::processEvents();

	image->updateTexture(tab, false);
}

unsigned int LayerSlice::width() const {
	return m_datasetS->width();
}
unsigned int LayerSlice::height() const {
	return m_datasetS->height();
}

unsigned int LayerSlice::depth() const {
	return m_datasetS->depth();
}

QUuid LayerSlice::seismicID() const {
	return m_datasetS->dataID();
}

QUuid LayerSlice::rgtID() const {
	return m_datasetT->dataID();
}

QVector2D LayerSlice::rgtMinMax(){
	return m_datasetT->minMax(m_channelT);
}

QUuid LayerSlice::dataID() const {
	return m_datasetS->dataID();
}

QString LayerSlice::name() const {
	return m_Name;
}

int LayerSlice::getSimplifyMeshSteps() const
{
	return m_simplifyMeshSteps;
}

void LayerSlice::setSimplifyMeshSteps(int steps)
{
	if(m_simplifyMeshSteps != steps)
	{
		m_simplifyMeshSteps = steps;
	}
}

int LayerSlice::getCompressionMesh() const
{
	return m_compressionMesh;
}

void LayerSlice::setCompressionMesh(int compress)
{
	if(m_compressionMesh != compress)
	{
		m_compressionMesh = compress;
	}
}

IGraphicRepFactory* LayerSlice::graphicRepFactory() {
	return m_repFactory.get();
}

void LayerSlice::setSeeds(const std::vector<RgtSeed>& seeds) {
	m_layerSpectrumProcess->setSeeds(seeds);
	m_morletProcess->setSeeds(seeds);
	m_gccProcess->setSeeds(seeds);
	m_tmapProcess->setSeeds(seeds);
	m_attributProcess->setSeeds(seeds);
	m_anisotropyProcess->setSeeds(seeds);
	this->m_seeds = seeds;
}

bool LayerSlice::getPolarity() const {
	return m_polarity;
}

void LayerSlice::setPolarity(bool polarity) {
	m_layerSpectrumProcess->setPolarity(polarity);
	m_morletProcess->setPolarity(polarity);
	m_gccProcess->setPolarity(polarity);
	m_tmapProcess->setPolarity(polarity);
	m_attributProcess->setPolarity(polarity);
	m_anisotropyProcess->setPolarity(polarity);
	this->m_polarity = polarity;
}

int LayerSlice::getDistancePower() const {
	return m_distancePower;
}

void LayerSlice::setDistancePower(int dist) {
	m_layerSpectrumProcess->setDistancePower(dist);
	m_morletProcess->setDistancePower(dist);
	m_gccProcess->setDistancePower(dist);
	m_tmapProcess->setDistancePower(dist);
	m_attributProcess->setDistancePower(dist);
	m_anisotropyProcess->setDistancePower(dist);
	m_distancePower = dist;
}

bool LayerSlice::getUseSnap() const {
	return m_useSnap;
}

void LayerSlice::setUseSnap(bool val) {
	m_layerSpectrumProcess->setUseSnap(val);
	m_morletProcess->setUseSnap(val);
	m_gccProcess->setUseSnap(val);
	m_tmapProcess->setUseSnap(val);
	m_attributProcess->setUseSnap(val);
	m_anisotropyProcess->setUseSnap(val);
	m_useSnap = val;
}

int LayerSlice::getSnapWindow() const {
	return m_snapWindow;
}

long LayerSlice::getDTauReference() const {
	return m_dtauReference;
}

void LayerSlice::setSnapWindow(int val) {
	m_layerSpectrumProcess->setSnapWindow(val);
	m_morletProcess->setSnapWindow(val);
	m_gccProcess->setSnapWindow(val);
	m_tmapProcess->setSnapWindow(val);
	m_attributProcess->setSnapWindow(val);
	m_anisotropyProcess->setSnapWindow(val);
	m_snapWindow = val;
}

bool LayerSlice::getUseMedian() const {
	return m_useMedian;
}

void LayerSlice::setUseMedian(bool val) {
	m_layerSpectrumProcess->setUseMedian(val);
	m_morletProcess->setUseMedian(val);
	m_gccProcess->setUseMedian(val);
	m_tmapProcess->setUseMedian(val);
	m_attributProcess->setUseMedian(val);
	m_anisotropyProcess->setUseMedian(val);
	m_useMedian = val;
}

int LayerSlice::getLWXMedianFilter() const {
	return m_lwx_medianFilter;
}

void LayerSlice::setLWXMedianFilter(int lwx) {
	m_layerSpectrumProcess->setLWXMedianFilter(lwx);
	m_morletProcess->setLWXMedianFilter(lwx);
	m_gccProcess->setLWXMedianFilter(lwx);
	m_tmapProcess->setLWXMedianFilter(lwx);
	m_attributProcess->setLWXMedianFilter(lwx);
	m_anisotropyProcess->setLWXMedianFilter(lwx);
	m_lwx_medianFilter = lwx;
}

int LayerSlice::getWindowSize() const {
	return m_windowSize;
}

int LayerSlice::getGccOffset() const {
	return m_gccOffset;
}

float LayerSlice::getHatPower() const {
	return m_hatPower;
}

void LayerSlice::setFreqMax(int freqMax) {
	m_morletProcess->setFreqMax(freqMax);
	m_freqMax = freqMax;
}

void LayerSlice::setFreqMin(int freqMin = 20) {
	m_morletProcess->setFreqMin(freqMin);
	m_freqMin = freqMin;
}

void LayerSlice::setFreqStep(int freqStep = 2) {
	m_morletProcess->setFreqStep(freqStep);
	m_freqStep = freqStep;
}

void LayerSlice::setW(int w) {
	m_gccProcess->setW(w);
	m_w = w;
}

void LayerSlice::setShift(int shift) {
	m_gccProcess->setShift(shift);
	m_shift = shift;
}

int LayerSlice::getMethod() const {
	return m_method;
}

void LayerSlice::setMethod(int method = 0) {
	m_method = method;
}

void LayerSlice::setType(int type)
{
	m_type = type;
}

void LayerSlice::setWindowSize(int windowSize) {
	m_layerSpectrumProcess->setWindowSize(windowSize);
	m_attributProcess->setWindowSize(windowSize);
	m_anisotropyProcess->setWindowSize(windowSize);
	this->m_windowSize = windowSize;
}

void LayerSlice::setGccOffset(int gccOffset) {
	m_gccProcess->setGccOffset(gccOffset);
	this->m_gccOffset = gccOffset;
}

void LayerSlice::setHatPower(float hatPower) {
	m_layerSpectrumProcess->setHatPower(hatPower);
	m_hatPower = hatPower;
}

void LayerSlice::setDTauReference(long dtau) {
	m_layerSpectrumProcess->setDTauReference(dtau);
	m_gccProcess->setDTauReference(dtau);
	m_morletProcess->setDTauReference(dtau);
	m_tmapProcess->setDTauReference(dtau);
	m_attributProcess->setDTauReference(dtau);
	m_anisotropyProcess->setDTauReference(dtau);
	m_dtauReference = dtau;
}

double LayerSlice::getFrequency(long fIdx) const {
	return getFrequencyStatic(fIdx, seismic()->sampleTransformation()->a(), getWindowSize());
}

double LayerSlice::getFrequencyStatic(long freqIndex, double pasech, long windowSize)  {
	return 1000/(pasech*(windowSize-1)) * (freqIndex+1);
}

int LayerSlice::getTmapExampleSize() const {
	return m_tmapExampleSize;
}

void LayerSlice::setTmapExampleSize(int val) {
	m_tmapProcess->setTmapExampleSize(val);
	m_tmapExampleSize = val;
}

int LayerSlice::getTmapSize() const {
	return m_tmapSize;
}

void LayerSlice::setTmapSize(int val) {
	m_tmapProcess->setTmapSize(val);
	m_tmapSize = val;
}

int LayerSlice::getTmapExampleStep() const {
	return m_tmapExampleStep;
}

void LayerSlice::setTmapExampleStep(int val) {
	m_tmapProcess->setTmapExampleStep(val);
	m_tmapExampleStep = val;
}


// JDTODO
void LayerSlice::computeProcess(LayerSpectrumDialog *layerspectrumdialog) {
	if (m_method == 0) {
		if (m_morletProcess)
			m_morletProcess->compute(layerspectrumdialog);
	} else if (m_method == 1) {
		if (m_layerSpectrumProcess)
			m_layerSpectrumProcess->compute(layerspectrumdialog);
	} else if (m_method == 2 ) {
		if (m_gccProcess)
		{
			m_gccProcess->compute(layerspectrumdialog);
		}
	} else if (m_method == 3) {
		if (m_tmapProcess) {
			m_tmapProcess->compute(layerspectrumdialog);
		}
	} else if (m_method == 4) {
		if (m_attributProcess) {
			m_attributProcess->compute(layerspectrumdialog);
		}
	} else if (m_method == 5) {
		if (m_anisotropyProcess) {
			m_anisotropyProcess->compute(layerspectrumdialog);
		}
	}
	/*
	else if( m_method > 2 && m_method < 25 )
	{
		if ( m_attributProcess )
		{
			m_attributProcess->compute(layerspectrumdialog);
		}
	}
	*/

}

LayerSlice::~LayerSlice() {
	// TODO Auto-generated destructor stub
	for(QMetaObject::Connection con : m_conn){
		QObject::disconnect(con);
	}
}

const float* LayerSlice::getModuleData(unsigned int freq) {
	const float* tab = nullptr;
	if (m_method == 0) {
		if (m_morletProcess)
			tab = m_morletProcess->getModuleData(freq);
	} else if (m_method == 1) {
		if (m_layerSpectrumProcess)
			tab = m_layerSpectrumProcess->getModuleData(freq);
	} else if (m_method == 2) {
		if (m_gccProcess)
			tab = m_gccProcess->getModuleData(freq);
	} else if (m_method == 3) {
		if (m_tmapProcess)
			tab = m_tmapProcess->getModuleData(freq);
	} else if (m_method == 4) {
		if (m_attributProcess)
			tab = m_attributProcess->getModuleData(freq);
	} else if (m_method == 5) {
		if (m_anisotropyProcess)
			tab = m_anisotropyProcess->getModuleData(freq);
	}
	return tab;
}

unsigned int LayerSlice::getNbOutputSlices() const {
	unsigned int nbOutputSlice = 0;
	if (m_method == 0) {
		if (m_morletProcess)
			nbOutputSlice = m_morletProcess->getNbOutputSlices();
	} else if (m_method == 1) {
		if (m_layerSpectrumProcess)
			nbOutputSlice = m_layerSpectrumProcess->getNbOutputSlices();
	} else if (m_method == 2) {
		if (m_gccProcess)
			nbOutputSlice = m_gccProcess->getNbOutputSlices();
	} else if (m_method == 3) {
		if (m_tmapProcess)
			nbOutputSlice = m_tmapProcess->getNbOutputSlices();
	} else if (m_method == 4) {
		if (m_attributProcess)
			nbOutputSlice = m_attributProcess->getNbOutputSlices();
	} else if (m_method == 5) {
		if (m_anisotropyProcess)
			nbOutputSlice = m_anisotropyProcess->getNbOutputSlices();
	}
	return nbOutputSlice;
}

bool LayerSlice::isModuleComputed() const {
	bool isModuleComputed = false;
	if (m_method == 0) {
		if (m_morletProcess)
			isModuleComputed = m_morletProcess->isModuleComputed();
	} else if (m_method == 1) {
		if (m_layerSpectrumProcess)
			isModuleComputed = m_layerSpectrumProcess->isModuleComputed();
	} else if (m_method == 2) {
		if (m_gccProcess)
			isModuleComputed = m_gccProcess->isModuleComputed();
	} else if (m_method == 3) {
		if (m_tmapProcess)
			isModuleComputed = m_tmapProcess->isModuleComputed();
	} else if (m_method == 4) {
		if (m_attributProcess)
			isModuleComputed = m_attributProcess->isModuleComputed();
	} else if (m_method == 5) {
		if (m_anisotropyProcess)
			isModuleComputed = m_anisotropyProcess->isModuleComputed();
	}
	return isModuleComputed;
}

void LayerSlice::setConstrainLayer(FixedLayerFromDataset* layer, QString isoName) {
    std::vector<float> vectIso;
    if (layer!=nullptr) {
            vectIso.resize(static_cast<long>(layer->getNbTraces())*layer->getNbProfiles(), 0);
            bool isValid = layer->readProperty(vectIso.data(), isoName);
            if (!isValid) {
                    vectIso.clear();
            }
    }
    m_layerSpectrumProcess->setConstrainLayer(vectIso);
    m_morletProcess->setConstrainLayer(vectIso);
    m_gccProcess->setConstrainLayer(vectIso);
    m_tmapProcess->setConstrainLayer(vectIso);
    m_attributProcess->setConstrainLayer(vectIso);
    m_anisotropyProcess->setConstrainLayer(vectIso);
    m_constrainLayer = layer;
    m_constrainIsoName = isoName;

}


void LayerSlice::setReferenceLayers(const std::vector<std::shared_ptr<FixedLayerFromDataset>>& layer,
			QString isoName, QString tauName) {
	std::vector<ReferenceDuo> vect;
	ReferenceDuo init;
	vect.resize(layer.size(), init);
	for (std::size_t index=0; index<layer.size(); index++) {
		ReferenceDuo& pair = vect[index];
		pair.iso.resize(static_cast<long>(layer[0]->getNbTraces())*layer[0]->getNbProfiles(), 0);
		pair.rgt.resize(static_cast<long>(layer[0]->getNbTraces())*layer[0]->getNbProfiles(), 0);
		bool isValid = layer[index]->readProperty(pair.iso.data(), isoName);
		isValid = isValid && layer[index]->readProperty(pair.rgt.data(), tauName);
		if (!isValid) {
			vect.clear();
			break;
		}
	}
	m_layerSpectrumProcess->setReferenceLayer(vect);
	m_morletProcess->setReferenceLayer(vect);
	m_gccProcess->setReferenceLayer(vect);
	m_tmapProcess->setReferenceLayer(vect);
	m_attributProcess->setReferenceLayer(vect);
	m_anisotropyProcess->setReferenceLayer(vect);
	m_referenceLayer = layer;
	m_referenceIsoName = isoName;
	m_referenceTauName = tauName;
}

void LayerSlice::setAttributDatasets(const QList<std::pair<Seismic3DDataset*, int>>& datasets) {
	if (m_method==4 && m_attributProcess) {
		m_datasets = datasets;
		m_attributProcess->setAttributDatasets(datasets);
	}
}

void LayerSlice::setDatasetsAndAngles(const QList<std::tuple<Seismic3DDataset*, int, float>>& datasetsAndAngles) {
	if (m_method==5 && m_anisotropyProcess) {
		m_datasetsAndAngles = datasetsAndAngles;
		m_anisotropyProcess->setAttributDatasets(datasetsAndAngles);
	}
}

QString LayerSlice::getLabelFromPosition(int val) const {
	QString label;
	if (val==0) {
		label = "Isochrone";
	} else if (val==1) {
		label = "Amplitude";
	} else if (val>=2 && val < getNbOutputSlices()) {
		if (m_method == 0) {
			float freqIndex = val - 2;
			float freq = getFrequency(freqIndex);
			//1000/(m_layerSlice->seismic()->sampleTransformation()->a()*(m_layerSlice->getWindowSize()-1)) * freqIndex;
			label = QString::number(freq)+" Hz";
		} else if (m_method == 1) {
			float freqIndex = val - 2;
			float freq = getFrequency(freqIndex);
			//1000/(m_layerSlice->seismic()->sampleTransformation()->a()*(m_layerSlice->getWindowSize()-1)) * freqIndex;
			label = QString::number(freq)+" Hz";
		} else if (m_method == 2) {
			int gccIndex = val - 1;
			int coherenceIndex = val - 1 - m_gccOffset;
			if (coherenceIndex>0) {
				label = "Coherence "+QString::number(coherenceIndex);
			} else {
				label = "GCC "+QString::number(gccIndex);
			}
		} else if (m_method == 3) {
			int index = val - 1;
			label = "Label " + QString::number(index);
		} else if (m_method == 4) {
			int index = val - 2;
			if (index<m_datasets.count()) {
				label = m_datasets[index].first->name();
			} else {
				label = "";
			}
		} else {
			if (val==2) {
				label = "Anisotropy";
			} else if (val==3) {
				label = "Angle";
			} else if (val==4) {
				label = "sqrt(L1)";
			} else if (val==5) {
				label = "sqrt(L2)";
			} else {
				label = "";
			}
		}
	}
	return label;
}
void LayerSlice::lockPalette(const QString& label, const PaletteParameters& params) {
	m_cachedPaletteParameters[label] = params;
}

void LayerSlice::unlockPalette(const QString& label) {
	m_cachedPaletteParameters.erase(label);
}

std::pair<bool, LayerSlice::PaletteParameters> LayerSlice::getLockedPalette(const QString& label) const {
	std::map<QString, PaletteParameters>::const_iterator it = std::find_if(m_cachedPaletteParameters.begin(),
			m_cachedPaletteParameters.end(), [label](const std::pair<QString, PaletteParameters>& pair) {
		return pair.first.compare(label)==0;
	});
	bool valid = it!=m_cachedPaletteParameters.end();
	PaletteParameters params;
	if (valid) {
		params = it->second;
	}
	return std::pair<bool, PaletteParameters>(valid, params);
}

void LayerSlice::deleteRep(){
	if(m_TreeDeletionProcess == false){
		m_TreeDeletionProcess = true;
		emit deletedMenu();
		this->workingSetManager()->deleteLayerSlice(this);
	}
}

void LayerSlice::setProcessDeletion(bool bValue){
	m_TreeDeletionProcess = bValue;
}

bool LayerSlice::ProcessDeletion() const{
	return m_TreeDeletionProcess;
}


IsoSurfaceBuffer LayerSlice::getIsoBuffer()
{
	IsoSurfaceBuffer res;

	res.buffer  = std::make_shared<CPUImagePaletteHolder>(width(), depth(),ImageFormats::QSampleType::FLOAT32, m_datasetS->ijToXYTransfo());

	//res.buffer = new CPUImagePaletteHolder(width(), depth(),ImageFormats::QSampleType::FLOAT32, m_datasetS->ijToXYTransfo());


	m_isoSurfaceHolder->lockPointer();

	void* tab = m_isoSurfaceHolder->backingPointer();

	QByteArray array(width()*depth()*sizeof(float),0);

	memcpy(array.data(), tab,width()*depth()*sizeof(float) );


	m_isoSurfaceHolder->unlockPointer();

	res.buffer->updateTexture(array, false);

	res.originSample = m_datasetS->sampleTransformation()->b();
	res.stepSample = m_datasetS->sampleTransformation()->a();



	return res;
}

ITreeWidgetItemDecorator* LayerSlice::getTreeWidgetItemDecorator() {
	if (m_decorator==nullptr) {
		m_decorator = new TextColorTreeWidgetItemDecorator(QColor(Qt::cyan), this);
	}
	return m_decorator;
}





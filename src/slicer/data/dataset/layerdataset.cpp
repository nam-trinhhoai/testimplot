#include "layerdataset.h"


#include <QFileInfo>
#include <QDebug>
#include <QString>
#include <iostream>
#include "Xt.h"
#include "cudaimagepaletteholder.h"
#include "slicerep.h"
#include "slicepositioncontroler.h"
#include <cuda.h>
#include "cuda_volume.h"
#include "cuda_algo.h"
#include <QRunnable>
#include <QThreadPool>
#include <QElapsedTimer>

#include "workingsetmanager.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "layerdatasetgrahicrepfactory.h"
#include  "cudadatasetminmaxtile.h"

#include "LayerSpectrumProcess.h"
#include "MorletProcess.h"
#include "GradientMultiScaleProcess.h"


#define TILE_SIZE  128

LayerDataset::LayerDataset(SeismicSurvey *survey,const QString &name,
		WorkingSetManager *workingSet,
		Seismic3DAbstractDataset *datasetS, Seismic3DAbstractDataset *datasetT,
		CUBE_TYPE type, QObject *parent) :
		Seismic3DAbstractDataset(survey,name, workingSet, type, parent)	{
	this->m_datasetS = datasetS;
	m_datasetT = datasetT;
	m_repFactory = new LayerDatasetGraphicRepFactory(this);
	m_currentFile = nullptr;
	m_headerLength = 0;

	//TODO WARNING pas propre: PROCESS instances should be created according to method used...
	// Ceci est de la dauble, mais il y a le feu!

	m_layerSpectrumProcess = new LayerSpectrumProcess<short>(
			datasetS, datasetT);
	m_morletProcess = new MorletProcess<short>(datasetS, datasetT);
	m_gccProcess = new GradientMultiScaleProcess<short>(datasetS, datasetT);
}

IGraphicRepFactory* LayerDataset::graphicRepFactory() {
	return m_repFactory;
}

void LayerDataset::loadFromXt(const std::string &path) {
	m_path=path;
	{
		inri::Xt xt(path.c_str());
		if (!xt.is_valid()) {
			std::cerr << "xt cube is not valid (" << path << ")" << std::endl;
			return;
		}
		m_height = xt.nSamples();
		m_width = xt.nRecords();
		m_depth = xt.nSlices();

		m_headerLength = (size_t)xt.header_size();
	}

	m_currentFile = fopen(path.c_str(), "rb");
	if (!m_currentFile) {
		fprintf(stderr, "Error opening file '%s'\n", path.c_str());
		return;
	}

	//initialize here a default transformation
	initializeTransformation();
}

void LayerDataset::readInlineBlock(void * output,int z0, int z1)
{
	QMutexLocker locker(&m_lock);
	unsigned int w = m_width;
	unsigned int h = m_height;

	QElapsedTimer timer;
	size_t absolutePosition = m_headerLength
								+ w * h * z0 * sizeof(short);
	fseek(m_currentFile, absolutePosition, SEEK_SET);
	fread(output, sizeof(short), w * h * (z1-z0), m_currentFile);
}

void LayerDataset::loadInlineXLine(CUDAImagePaletteHolder *image,
		SliceDirection dir, unsigned int z) {

	if (dir == SliceDirection::Inline) {

		class InlineRunnable: public QRunnable {
			LayerDataset *m_d;
			unsigned int m_pos;
			CUDAImagePaletteHolder *m_image;
		public:
			InlineRunnable(LayerDataset *e, CUDAImagePaletteHolder *image,
					unsigned int z) :
					QRunnable() {
				m_d = e;
				m_pos = z;
				m_image = image;
			}
			void run() {
				unsigned int w = m_d->m_width;
				unsigned int h = m_d->m_height;
				unsigned int d = m_d->m_depth;
				short tmp[w * h];
				{
					QMutexLocker locker(&m_d->m_lock);
					size_t absolutePosition = m_d->m_headerLength
							+ w * h * m_pos * sizeof(short);
					fseek(m_d->m_currentFile, absolutePosition, SEEK_SET);
					fread(tmp, sizeof(short), w * h, m_d->m_currentFile);
				}

				m_image->updateTexture(tmp, true);
			}
		};
		qDebug()<<"Loading slice:"<<z;
		InlineRunnable *r = new InlineRunnable(this, image, z);
		QThreadPool::globalInstance()->start(r);
	} else {
		class XlineRunnable: public QRunnable {
			LayerDataset *m_d;
			unsigned int m_pos;
			CUDAImagePaletteHolder *m_image;
		public:
			XlineRunnable(LayerDataset *e, CUDAImagePaletteHolder *image,
					unsigned int z) :
					QRunnable() {
				m_d = e;
				m_pos = z;
				m_image = image;
			}
			void run() {
				unsigned int w = m_d->m_width;
				unsigned int h = m_d->m_height;
				unsigned int d = m_d->m_depth;
				short temp[h * d];
				memset(temp, 0, h * d * sizeof(short));
				{
					QMutexLocker locker(&m_d->m_lock);
					fseek(m_d->m_currentFile,
							m_d->m_headerLength + m_pos * h * sizeof(short),
							SEEK_SET);

					size_t seekOffset = (h * w - h) * sizeof(short);
					int numTile = d / TILE_SIZE + 1;
					for (int i = 0; i < numTile; i++) {
						int d0 = i * TILE_SIZE;
						int d1 = d0 + TILE_SIZE;
						if (d1 > d)
							d1 = d;
						for (int k = d0; k < d1; k++) {
							fread(temp + k * h, sizeof(short), h,
									m_d->m_currentFile);
							fseek(m_d->m_currentFile, seekOffset, SEEK_CUR);
						}
						m_image->updateTexture(temp, true);
					}
				}

			}
		};
		XlineRunnable *r = new XlineRunnable(this, image, z);
		QThreadPool::globalInstance()->start(r);
	}
}

QVector2D LayerDataset::minMax(bool forced) {
	//By default dynamic is already defined
	if(!forced && m_type==CUBE_TYPE::RGT)
	{
		return QVector2D(0,32000);
	}
	if (m_internalMinMaxCache.initialized && !forced)
		return m_internalMinMaxCache.range;

	CUDADatasetMinMaxTile tile(width(),height(),TILE_SIZE);
	int numTile = depth() / TILE_SIZE + 1;
	QVector<QVector2D> tileCoords;
	float max=std::numeric_limits<short>::lowest();
	float min=std::numeric_limits<short>::max();
	for (int i = 0; i < numTile; i++) {
		int d0 = i * TILE_SIZE;
		int d1 = d0 + TILE_SIZE;
		if (d1 > depth())
			d1=depth();
		//TODO !!!!!!!!!!!!!!!!!!!!!!!!! QVector2D temp= tile.minMax(d0,d1,this);
		//TODO min=std::min(min,temp.x());
		//TODO max=std::max(max,temp.y());
	}
	m_internalMinMaxCache.range=QVector2D(min,max);
	m_internalMinMaxCache.initialized = true;
	std::cout<<min<<"\t"<<max<<std::endl;
	return m_internalMinMaxCache.range;
}

void LayerDataset::setSeeds(const std::vector<RgtSeed>& seeds) {
	m_layerSpectrumProcess->setSeeds(seeds);
	m_morletProcess->setSeeds(seeds);
	m_gccProcess->setSeeds(seeds);
	this->m_seeds = seeds;
}

bool LayerDataset::getPolarity() const {
	return m_polarity;
}

void LayerDataset::setPolarity(bool polarity) {
	m_layerSpectrumProcess->setPolarity(polarity);
	m_morletProcess->setPolarity(polarity);
	m_gccProcess->setPolarity(polarity);
	this->m_polarity = polarity;
}

int LayerDataset::getDistancePower() const {
	return m_distancePower;
}

void LayerDataset::setDistancePower(int dist) {
	m_layerSpectrumProcess->setDistancePower(dist);
	m_morletProcess->setDistancePower(dist);
	m_gccProcess->setDistancePower(dist);
	m_distancePower = dist;
}

bool LayerDataset::getUseSnap() const {
	return m_useSnap;
}

void LayerDataset::setUseSnap(bool val) {
	m_layerSpectrumProcess->setUseSnap(val);
	m_morletProcess->setUseSnap(val);
	m_gccProcess->setUseSnap(val);
	m_useSnap = val;
}

int LayerDataset::getSnapWindow() const {
	return m_snapWindow;
}

void LayerDataset::setSnapWindow(int val) {
	m_layerSpectrumProcess->setSnapWindow(val);
	m_morletProcess->setSnapWindow(val);
	m_gccProcess->setSnapWindow(val);
	m_snapWindow = val;
}

bool LayerDataset::getUseMedian() const {
	return m_useMedian;
}

void LayerDataset::setUseMedian(bool val) {
	m_layerSpectrumProcess->setUseMedian(val);
	m_morletProcess->setUseMedian(val);
	m_gccProcess->setUseMedian(val);
	m_useMedian = val;
}

int LayerDataset::getLWXMedianFilter() const {
	return m_lwx_medianFilter;
}

void LayerDataset::setLWXMedianFilter(int lwx) {
	m_layerSpectrumProcess->setLWXMedianFilter(lwx);
	m_morletProcess->setLWXMedianFilter(lwx);
	m_gccProcess->setLWXMedianFilter(lwx);
	m_lwx_medianFilter = lwx;
}

int LayerDataset::getWindowSize() const {
	return m_windowSize;
}

float LayerDataset::getHatPower() const {
	return m_hatPower;
}

void LayerDataset::setFreqMax(int freqMax) {
	m_morletProcess->setFreqMax(freqMax);
	m_freqMax = freqMax;
}

void LayerDataset::setFreqMin(int freqMin = 20) {
	m_morletProcess->setFreqMin(freqMin);
	m_freqMin = freqMin;
}

void LayerDataset::setFreqStep(int freqStep = 2) {
	m_morletProcess->setFreqStep(freqStep);
	m_freqStep = freqStep;
}

void LayerDataset::setW(int w) {
	m_gccProcess->setW(w);
	m_w = w;
}

void LayerDataset::setShift(int shift) {
	m_gccProcess->setShift(shift);
	m_shift = shift;
}

int LayerDataset::getMethod() const {
	return m_method;
}

void LayerDataset::setMethod(int method = 0) {
	m_method = method;
}

void LayerDataset::setWindowSize(int windowSize) {
	m_layerSpectrumProcess->setWindowSize(windowSize);
	m_gccProcess->setWindowSize(windowSize);
	this->m_windowSize = windowSize;
}

void LayerDataset::setHatPower(float hatPower) {
	m_layerSpectrumProcess->setHatPower(hatPower);
	m_hatPower = hatPower;
}

void LayerDataset::computeProcess() {
	if (m_method == 0)
		m_morletProcess->compute();
	else if (m_method == 1)
		m_layerSpectrumProcess->compute();
	else if (m_method == 2)
		m_gccProcess->compute();
}

LayerDataset::~LayerDataset() {
	if (m_currentFile)
		fclose(m_currentFile);
}


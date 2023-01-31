#include "computationoperatordatasetrep.h"
#include "cudaimagepaletteholder.h"
#include "computationoperatordatasetproppanel.h"
#include "computationoperatordatasetlayer.h"
#include "qgllineitem.h"
#include "slicepositioncontroler.h"
#include "qglisolineitem.h"
#include "computationoperatordataset.h"
#include "abstractinnerview.h"
#include "workingsetmanager.h"
#include "GraphEditor_PolygonShape.h"
#include "GraphEditor_EllipseShape.h"
#include "GraphEditor_RectShape.h"
#include "qglfullcudaimageitem.h"

#include <iostream>
#include <QGraphicsScene>
#include <QAction>
#include <QMenu>
#include <QtGlobal>

ComputationOperatorDatasetRep::ComputationOperatorDatasetRep(ComputationOperatorDataset *data,
		CUDAImagePaletteHolder *attributeHolder, const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo,
		SliceDirection dir, AbstractInnerView *parent) :
	  AbstractGraphicRep(parent), IMouseImageDataProvider() {
	m_data = data;
	m_channel = 0;
	m_sliceRangeAndTransfo = sliceRangeAndTransfo;
	m_currentSlice = 0;
	m_dir = dir;

	m_image = attributeHolder;

	m_propPanel = nullptr;
	m_layer = nullptr;
	m_showColorScale = false;

	m_name=m_data->name();

	//m_cache.resize(m_image->width() * m_image->height() * m_data->dimV() * m_data->sampleType().byte_size());
	m_cache.reset(m_data->createInlineXLineCache(m_dir));

	connect(m_image, SIGNAL(dataChanged()), this, SLOT(dataChanged()));
	connect(m_data, SIGNAL(rangeLockChanged()), this, SLOT(rangeLockChanged()));
}

void ComputationOperatorDatasetRep::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
}

IData* ComputationOperatorDatasetRep::data() const {
	return m_data;
}

QPair<QVector2D,AffineTransformation>  ComputationOperatorDatasetRep::sliceRangeAndTransfo() const {
	return m_sliceRangeAndTransfo;
}

void ComputationOperatorDatasetRep::showColorScale(bool val) {
	m_showColorScale = val;
	m_layer->showColorScale(m_showColorScale);
}
bool ComputationOperatorDatasetRep::colorScale() const {
	return m_showColorScale;
}

QWidget* ComputationOperatorDatasetRep::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new ComputationOperatorDatasetPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* ComputationOperatorDatasetRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		loadSlice(m_currentSlice);
		m_layer = new ComputationOperatorDatasetLayer(this, scene, defaultZDepth, parent);
		m_layer->showColorScale(m_showColorScale);
		connect(m_layer, &ComputationOperatorDatasetLayer::hidden, this, &ComputationOperatorDatasetRep::relayHidden);
	}
	return m_layer;
}

void ComputationOperatorDatasetRep::refreshLayer() {
	if (m_layer == nullptr)
		return;

	m_layer->refresh();
}

ComputationOperatorDatasetRep::~ComputationOperatorDatasetRep() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;
	for (QGraphicsItem* item : m_datacontrolers) {
		if (item!=nullptr && item->scene()!=nullptr) {
			item->scene()->removeItem(item);
			delete item;
		} else if (item!=nullptr) {
			delete item;
		}
	}
}

int ComputationOperatorDatasetRep::currentSliceWorldPosition() const {
	double val;
	m_sliceRangeAndTransfo.second.direct((double)m_currentSlice,val);
	return (int)val;
}

int ComputationOperatorDatasetRep::currentSliceIJPosition() const {
	return m_currentSlice;
}

void ComputationOperatorDatasetRep::setSliceIJPosition(int val) {
	setSliceIJPosition(val, false);
}

void ComputationOperatorDatasetRep::setSliceWorldPosition(int val,bool force) {
	double imagePositionD;
	m_sliceRangeAndTransfo.second.indirect((double)val, imagePositionD);


	// avoid to go outside the bounds

	if (imagePositionD<0) {
		imagePositionD = 0;
		double _val;
		m_sliceRangeAndTransfo.second.direct(imagePositionD, _val);
		val = _val;
	} else if (m_dir==SliceDirection::Inline && imagePositionD>=m_data->depth()) {
		imagePositionD = m_data->depth() - 1;
		double _val;
		m_sliceRangeAndTransfo.second.direct(imagePositionD, _val);
		val = _val;
	} else if (m_dir==SliceDirection::XLine && imagePositionD>=m_data->width()) {
		imagePositionD = m_data->width() - 1;
		double _val;
		m_sliceRangeAndTransfo.second.direct(imagePositionD, _val);
		val = _val;
	}

	int pos=(int)imagePositionD;
	if(m_currentSlice==pos && !force)
		return;

	loadSlice(pos);
	refreshLayer();

	emit sliceWordPositionChanged(val);
	emit sliceIJPositionChanged(pos);
}

void ComputationOperatorDatasetRep::setSliceIJPosition(int val,bool force) {
	if(m_currentSlice==val && !force)
		return;
	loadSlice(val);
	refreshLayer();

	double imagePositionD;
	m_sliceRangeAndTransfo.second.direct((double)val, imagePositionD);
	int pos=(int)imagePositionD;

	emit sliceIJPositionChanged(val);
	emit sliceWordPositionChanged(pos);
}


void ComputationOperatorDatasetRep::loadSlice(unsigned int z) {
	m_currentSlice = z;
	m_cacheMutex.lock();
	m_data->loadInlineXLine(m_image, m_dir, z, m_channel, m_cache.get());//(void*) m_cache.data());
	m_cacheMutex.unlock();
}

bool ComputationOperatorDatasetRep::mouseData(double x, double y, MouseInfo &info) {
	double value;
	bool valid = IGeorefImage::value(m_image, x, y, info.i, info.j, value);
	info.valuesDesc.push_back("Image");
	info.values.push_back( value);
	info.depthValue = true;
	info.depth=y;
	info.depthUnit = m_data->cubeSeismicAddon().getSampleUnit();
	return valid;
}

bool ComputationOperatorDatasetRep::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> ComputationOperatorDatasetRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString ComputationOperatorDatasetRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

int ComputationOperatorDatasetRep::channel() const {
	return m_channel;
}

void ComputationOperatorDatasetRep::setChannel(int channel) {
	m_channel = channel;
	loadSlice(m_currentSlice);
	emit channelChanged(channel);
}

const std::vector<std::vector<char>>& ComputationOperatorDatasetRep::lockCache() {
	m_cacheMutex.lock();
	return m_cache->buffer();
}

void ComputationOperatorDatasetRep::unlockCache() {
	m_cacheMutex.unlock();
}

void ComputationOperatorDatasetRep::rangeLockChanged() {
	if (m_data->isRangeLocked()) {
		m_image->setRange(m_data->lockedRange());
	}
}

void ComputationOperatorDatasetRep::deleteGraphicItemDataContent(QGraphicsItem *item)
{
	deleteData(m_image,item);
}

AbstractGraphicRep::TypeRep ComputationOperatorDatasetRep::getTypeGraphicRep() {
    return AbstractGraphicRep::ImageRgt;
}

void ComputationOperatorDatasetRep::relayHidden() {
   emit layerHidden();
}

QGraphicsObject* ComputationOperatorDatasetRep::cloneCUDAImageWithMask(QGraphicsItem *parent)
{
	return new QGLFullCUDAImageItem(image(),parent,true);
}

#include "slicerep.h"
#include "cudaimagepaletteholder.h"
#include "sliceproppanel.h"
#include "slicelayer.h"
#include "qgllineitem.h"
#include "slicepositioncontroler.h"
#include "qglisolineitem.h"
#include "seismic3dabstractdataset.h"
#include "abstractinnerview.h"
#include "workingsetmanager.h"
#include "seismicsurvey.h"
#include "GraphEditor_PolygonShape.h"
#include "GraphEditor_EllipseShape.h"
#include "GraphEditor_RectShape.h"
#include "qglfullcudaimageitem.h"

#include <iostream>
#include <QGraphicsScene>
#include <QAction>
#include <QMenu>
#include <QtGlobal>

SliceRep::SliceRep(Seismic3DAbstractDataset *data,
		CUDAImagePaletteHolder *attributeHolder, const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo,
		SliceDirection dir, AbstractInnerView *parent) :
	  AbstractGraphicRep(parent), IDataControlerHolder(), IDataControlerProvider(), IMouseImageDataProvider() {
	m_data = data;
	m_channel = 0;
	m_sliceRangeAndTransfo = sliceRangeAndTransfo;
	m_currentSlice = 0;
	m_dir = dir;

	m_image = attributeHolder;

	m_propPanel = nullptr;
	m_layer = nullptr;
	m_controler = nullptr;
	m_showColorScale = false;

	m_name=m_data->name();

	//m_cache.resize(m_image->width() * m_image->height() * m_data->dimV() * m_data->sampleType().byte_size());
	m_cache.reset(m_data->createInlineXLineCache(m_dir));

	connect(m_image, SIGNAL(dataChanged()), this, SLOT(dataChanged()));
	connect(m_data, SIGNAL(rangeLockChanged()), this, SLOT(rangeLockChanged()));

	// MZR 17082021
	m_data->addRep(this);
	connect(m_data,SIGNAL(deletedMenu()),this,SLOT(deleteSliceRep()));
}

// MZR 14072021
void SliceRep::buildContextMenu(QMenu *menu){
	QAction *deleteAction = new QAction(tr("Unselect seismic"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteSliceRep()));
}

void SliceRep::deleteSliceRep(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	WorkingSetManager *manager = const_cast<WorkingSetManager*>(m_data->workingSetManager());
	SeismicSurvey* survey = const_cast<SeismicSurvey*>(m_data->survey());
	QList<Seismic3DAbstractDataset*> list = survey->datasets();
	for (int i = 0; i < list.size(); ++i) {
		if (list.at(i)->name()== m_name){
			Seismic3DAbstractDataset* dataSet = list.at(i);
			dataSet->deleteRep(this);
			if(dataSet->getTreeDeletionProcess() == false){
				dataSet->deleteRep();
			}

			if(dataSet->getRepListSize() == 0)
			    survey->removeDataset(dataSet);

			break;
		}
	}

	this->deleteLater();
}

void SliceRep::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
}

IData* SliceRep::data() const {
	return m_data;
}

QPair<QVector2D,AffineTransformation>  SliceRep::sliceRangeAndTransfo() const {
	return m_sliceRangeAndTransfo;
}

void SliceRep::showColorScale(bool val) {
	m_showColorScale = val;
	m_layer->showColorScale(m_showColorScale);
}
bool SliceRep::colorScale() const {
	return m_showColorScale;
}

QGraphicsItem* SliceRep::getOverlayItem(DataControler *c,
		QGraphicsItem *parent) {
	//A section representation can be controled only by datasets with the same id
	if (c->dataID() != m_data->dataID())
		return nullptr;

	if (SlicePositionControler *controler =
			dynamic_cast<SlicePositionControler*>(c)) {
		if (controler->direction() == m_dir)
			return nullptr;

		//The controler is the one of the survey
		QGLLineItem *item = new QGLLineItem(image()->worldExtent(),nullptr,
				QGLLineItem::Direction::VERTICAL, parent);
		item->updatePosition(controler->position());
		item->setColor(controler->color());

		connect(controler, SIGNAL(posChanged(int)), item,
				SLOT(updatePosition(int)));

		connect(item, SIGNAL(positionChanged(int)), controler,
				SLOT(requestPosChanged(int)));

		m_datacontrolers.insert(c, item);
		return item;
	}
	return nullptr;
}

void SliceRep::notifyDataControlerMouseMoved(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {
	QMap<DataControler*, QGraphicsItem*>::const_iterator i =
			m_datacontrolers.constBegin();
	while (i != m_datacontrolers.constEnd()) {
		if (QGLLineItem *item = dynamic_cast<QGLLineItem*>(i.value())) {
			item->mouseMoved(worldX, worldY, button, keys);
		}
		++i;
	}
}
void SliceRep::notifyDataControlerMousePressed(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {
	QMap<DataControler*, QGraphicsItem*>::const_iterator i =
			m_datacontrolers.constBegin();
	while (i != m_datacontrolers.constEnd()) {
		if (QGLLineItem *item = dynamic_cast<QGLLineItem*>(i.value())) {
			item->mousePressed(worldX, worldY, button, keys);
		}
		++i;
	}
}
void SliceRep::notifyDataControlerMouseRelease(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {
	QMap<DataControler*, QGraphicsItem*>::const_iterator i =
			m_datacontrolers.constBegin();
	while (i != m_datacontrolers.constEnd()) {
		if (QGLLineItem *item = dynamic_cast<QGLLineItem*>(i.value())) {
			item->mouseRelease(worldX, worldY, button, keys);
		}
		++i;
	}
}

void SliceRep::notifyDataControlerMouseDoubleClick(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {
}

QGraphicsItem* SliceRep::releaseOverlayItem(DataControler *controler) {

	if (!m_datacontrolers.contains(controler))
		return nullptr;

	QGraphicsItem *o = m_datacontrolers[controler];
	if (QGLLineItem *item = dynamic_cast<QGLLineItem*>(o)) {
		disconnect(controler, SIGNAL(posChanged(int)), item,
				SLOT(updatePosition(int)));

		disconnect(item, SIGNAL(positionChanged(int)), controler,
				SLOT(requestPosChanged(int)));
	} else if (QGLIsolineItem *item = dynamic_cast<QGLIsolineItem*>(o)) {
		disconnect(this, SIGNAL(sliceIJPositionChanged(int)), item,
				SLOT(updateSlice(int)));

		disconnect(controler, SIGNAL(extractionWindowChanged(uint)), item,
				SLOT(updateWindowSize(uint)));

		disconnect(controler, SIGNAL(rgtPosChanged(int)), item,
				SLOT(updateRGTPosition()));
	}
	m_datacontrolers.remove(controler);
	return o;
}

QWidget* SliceRep::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new SlicePropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* SliceRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new SliceLayer(this, scene, defaultZDepth, parent);
		m_layer->showColorScale(m_showColorScale);
		connect(m_layer, &SliceLayer::hidden, this, &SliceRep::relayHidden);
	}
	return m_layer;
}

void SliceRep::refreshLayer() {
	if (m_layer == nullptr)
		return;

	m_layer->refresh();
}

SliceRep::~SliceRep() {
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
	m_data->deleteRep(this);
}

int SliceRep::currentSliceWorldPosition() const {
	double val;
	m_sliceRangeAndTransfo.second.direct((double)m_currentSlice,val);
	return (int)val;
}

int SliceRep::currentSliceIJPosition() const {
	return m_currentSlice;
}

void SliceRep::setSliceWorldPosition(int val,bool force) {
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

void SliceRep::setSliceIJPosition(int val,bool force) {
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


void SliceRep::loadSlice(unsigned int z) {
	m_currentSlice = z;
	m_cacheMutex.lock();
	m_data->loadInlineXLine(m_image, m_dir, z, m_channel, m_cache.get());//(void*) m_cache.data());
	m_cacheMutex.unlock();
}

void SliceRep::setDataControler(DataControler *controler) {
	m_controler = controler;
}
DataControler* SliceRep::dataControler() const {
	return m_controler;
}

bool SliceRep::mouseData(double x, double y, MouseInfo &info) {
	double value;
	bool valid = IGeorefImage::value(m_image, x, y, info.i, info.j, value);
	info.valuesDesc.push_back("Image");
	info.values.push_back( value);
	info.depthValue = true;
	info.depth=y;
	info.depthUnit = m_data->cubeSeismicAddon().getSampleUnit();
	return valid;
}

bool SliceRep::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> SliceRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString SliceRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

int SliceRep::channel() const {
	return m_channel;
}

void SliceRep::setChannel(int channel) {
	m_channel = channel;
	loadSlice(m_currentSlice);
	emit channelChanged(channel);
}

const std::vector<char>& SliceRep::lockCache() {
	m_cacheMutex.lock();
	return m_cache->buffer();
}

void SliceRep::unlockCache() {
	m_cacheMutex.unlock();
}

void SliceRep::rangeLockChanged() {
	if (m_data->isRangeLocked()) {
		m_image->setRange(m_data->lockedRange());
	}
}

void SliceRep::deleteGraphicItemDataContent(QGraphicsItem *item)
{
	deleteData(m_image,item);
}

AbstractGraphicRep::TypeRep SliceRep::getTypeGraphicRep() {
	if (m_data->type()==Seismic3DAbstractDataset::CUBE_TYPE::RGT) {
		return AbstractGraphicRep::ImageRgt;
	} else {
		 return AbstractGraphicRep::Image;
	}
}

void SliceRep::relayHidden() {
   emit layerHidden();
}

QGraphicsObject* SliceRep::cloneCUDAImageWithMask(QGraphicsItem *parent)
{
	return new QGLFullCUDAImageItem(image(),parent,true);
}

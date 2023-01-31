#include "dataset3Dslicerep.h"
#include "cudaimagepaletteholder.h"
#include "seismic3dabstractdataset.h"
#include "dataset3Dslicelayer.h"
#include "dataset3Dproppanel.h"
#include <iostream>
#include <QMutexLocker>
#include "abstractinnerview.h"
#include "cubeseismicaddon.h"

#include <QCoreApplication>
#include <QMenu>
#include <QAction>

Dataset3DSliceRep::Dataset3DSliceRep(Seismic3DAbstractDataset *data,
		CUDAImagePaletteHolder *attributeHolder, const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo,
		SliceDirection dir, AbstractInnerView *parent) :
	  AbstractGraphicRep(parent){
	m_data = data;
	m_channel = 0;
	m_sliceRangeAndTransfo = sliceRangeAndTransfo;

	m_dir = dir;

	m_propPanel = nullptr;
	m_layer = nullptr;
	m_image = attributeHolder;

	connect(m_image, SIGNAL(dataChanged()), this, SLOT(dataChanged()));
	m_currentSlice = 0;
	m_name=generateName();
	loadSlice(0);
}
QString Dataset3DSliceRep::generateName()
{
	if(m_dir==SliceDirection::Inline)
	{
		return QString("Inline ")+QString::number(currentSliceWorldPosition());
	}else
		return QString("Xline ")+QString::number(currentSliceWorldPosition());
}
void Dataset3DSliceRep::dataChanged() {
	//We make the synchro at this level as we need to keep this order
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
}

int Dataset3DSliceRep::currentSliceWorldPosition() const {
	double val;
	m_sliceRangeAndTransfo.second.direct((double)m_currentSlice,val);
	return (int)val;
}

int Dataset3DSliceRep::currentSliceIJPosition() const {
	return m_currentSlice;
}

int Dataset3DSliceRep::channel() const {
	return m_channel;
}

IData* Dataset3DSliceRep::data() const {
	return m_data;
}

QPair<QVector2D,AffineTransformation>  Dataset3DSliceRep::sliceRangeAndTransfo() const {
	return m_sliceRangeAndTransfo;
}


QWidget* Dataset3DSliceRep::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new Dataset3DPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}


Graphic3DLayer * Dataset3DSliceRep::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera){
	if (m_layer == nullptr) {
		m_layer = new Dataset3DSliceLayer(this,parent,root,camera);
	}
	return m_layer;
}

Dataset3DSliceRep::~Dataset3DSliceRep() {
	if (m_layer!=nullptr) {
		delete m_layer;
	}
	if (m_propPanel!=nullptr) {
		delete m_propPanel;
	}
}

void Dataset3DSliceRep::setSliceWorldPosition(int val) {
	double imagePositionD;
	m_sliceRangeAndTransfo.second.indirect((double)val, imagePositionD);
	int valIJ = (int)imagePositionD;
	setSlicePositionInternal(val, valIJ);
}

void Dataset3DSliceRep::setSlicePosition(int valWorld, int valIJ) {
	if (m_writeMutex.tryLock()) {
		{
			QMutexLocker locker(&m_nextMutex);
			m_isNextDefined = false;
		}
		bool goOn = true;
		int valWorld = valWorld;
		int valIJ = valIJ;
		while(goOn) {
			setSlicePositionInternal(valWorld, valIJ);
			QCoreApplication::processEvents();
			QMutexLocker locker(&m_nextMutex);
			if (m_isNextDefined) {
				valWorld = m_nextWorldPos;
				valIJ = m_nextIJPos;
				m_isNextDefined = false;
			}
		}
		m_writeMutex.unlock();
	} else {
		QMutexLocker locker(&m_nextMutex);
		m_nextWorldPos = valWorld;
		m_nextIJPos = valIJ;
		m_isNextDefined = true;
	}
}

void Dataset3DSliceRep::setSlicePositionInternal(int valWorld, int valIJ) {
	if(m_currentSlice==valIJ)
			return;

	loadSlice(valIJ);

	setName(generateName());

	emit sliceWordPositionChanged(valWorld);
	emit sliceIJPositionChanged(valIJ);
}




void Dataset3DSliceRep::setSliceIJPosition(int val) {
	double imagePositionD;
	m_sliceRangeAndTransfo.second.direct((double)val, imagePositionD);
	int pos=(int)imagePositionD;
	setSlicePositionInternal(pos, val);
}
/*
void Dataset3DSliceRep::setSliceIJPositionInternal(int val) {
	if(m_currentSlice==val)
		return;
	loadSlice(val);

	double imagePositionD;
	m_sliceRangeAndTransfo.second.direct((double)val, imagePositionD);
	int pos=(int)imagePositionD;

	setName(generateName());

	emit sliceIJPositionChanged(val);
	emit sliceWordPositionChanged(pos);
}*/

void Dataset3DSliceRep::loadSlice(unsigned int z) {
	m_currentSlice = z;
	m_data->loadInlineXLine(m_image, m_dir, z, m_channel);
}

void Dataset3DSliceRep::setChannel(int channel) {
	m_channel = channel;
	loadSlice(m_currentSlice);
	emit channelChanged(m_channel);
}

bool Dataset3DSliceRep::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> Dataset3DSliceRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString Dataset3DSliceRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

void Dataset3DSliceRep::delete3DRep(){
	m_parent->hideRep(this);
    emit deletedRep(this);

    this->deleteLater();
}

AbstractGraphicRep::TypeRep Dataset3DSliceRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image3D;
}

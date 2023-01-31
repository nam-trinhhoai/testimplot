#include "seismicsurveyreponmap.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismicsurveyproppanel.h"
#include "seismicsurveylayer.h"
#include "slicepositioncontroler.h"
#include "affine2dtransformation.h"
#include "qgllineitem.h"
//#include "abstractinnerview.h"
#include "abstract2Dinnerview.h"
#include "cameraparameterscontroller.h"
#include <cmath>
#include <QSvgRenderer>
#include <QPen>

#include <QGraphicsView>
#include <QScreen>


#include <QGraphicsScene>

SeismicSurveyRepOnMap::SeismicSurveyRepOnMap(SeismicSurvey *survey, AbstractInnerView *parent) :
	SeismicSurveyRep(survey,parent),IDataControlerHolder() {
	m_propPanel = nullptr;
	m_layer = nullptr;
}

SeismicSurveyRepOnMap::~SeismicSurveyRepOnMap() {
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

Seismic3DAbstractDataset * SeismicSurveyRepOnMap::containsDatasetID(const QUuid &uuid)
{
	for(Seismic3DAbstractDataset * dataset:m_survey->datasets())
	{
		if(dataset->dataID()==uuid)
			return dataset;
	}
	return nullptr;
}

QGraphicsItem* SeismicSurveyRepOnMap::getOverlayItem(DataControler *c,QGraphicsItem *parent) {

	Seismic3DAbstractDataset *dataset=containsDatasetID(c->dataID());

	if(dataset==nullptr)
		return nullptr;

	if (SlicePositionControler *controler =
			dynamic_cast<SlicePositionControler*>(c)) {
		QGLLineItem::Direction dir = QGLLineItem::Direction::HORIZONTAL;
		if (controler->direction() == SliceDirection::XLine)
			dir = QGLLineItem::Direction::VERTICAL;

		QGLLineItem *item = new QGLLineItem(dataset->inlineXlineExtent(),m_survey->inlineXlineToXYTransfo(), dir,parent);
		item->updatePosition(controler->position());
		item->setColor(controler->color());

		connect(controler, SIGNAL(posChanged(int)), item,
				SLOT(updatePosition(int)));

		connect(item, SIGNAL(positionChanged(int)), controler,
						SLOT(requestPosChanged(int)));

		m_datacontrolers.insert(c,item);
		return item;
	}
	else if(CameraParametersController *controler = dynamic_cast<CameraParametersController*>(c))
	{
		//m_debug = true;
		HelicoItem* helicoItem = new HelicoItem(controler,parent);

		helicoItem->showHelico(controler->helicoVisible());
		m_datacontrolers.insert(c,helicoItem);

		Abstract2DInnerView * innerView = dynamic_cast<Abstract2DInnerView*>(m_parent);
		connect(innerView,&Abstract2DInnerView::viewAreaChanged, helicoItem,&HelicoItem::refreshItemZoomScale);

		return helicoItem;



	}
	return nullptr;
}


bool SeismicSurveyRepOnMap::eventFilter(QObject* watched, QEvent* ev) {
	if (ev->type() == QEvent::Wheel) {
		//transformItemZoomScale(m_item,m_ctrl,m_ctrl->position());
	}
	return false;
}


void SeismicSurveyRepOnMap::notifyDataControlerMouseMoved(double worldX, double worldY,
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
void SeismicSurveyRepOnMap::notifyDataControlerMousePressed(double worldX, double worldY,
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
void SeismicSurveyRepOnMap::notifyDataControlerMouseRelease(double worldX, double worldY,
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

void SeismicSurveyRepOnMap::notifyDataControlerMouseDoubleClick(double worldX, double worldY,
                Qt::MouseButton button, Qt::KeyboardModifiers keys) {
}

QGraphicsItem  *SeismicSurveyRepOnMap::releaseOverlayItem(DataControler *controler) {
	if(!m_datacontrolers.contains(controler))
			return nullptr;
	QGraphicsItem * o= m_datacontrolers[controler];

	if (QGLLineItem *item = dynamic_cast<QGLLineItem*>(o)) {
		disconnect(controler, SIGNAL(posChanged(int)), item,
				SLOT(updatePosition(int)));

		disconnect(item, SIGNAL(positionChanged(int)), controler,
								SLOT(requestPosChanged(int)));

	} else if (HelicoItem *item = dynamic_cast<HelicoItem*>(o)) {
		Abstract2DInnerView * innerView = dynamic_cast<Abstract2DInnerView*>(m_parent);
		if (innerView!=nullptr) {
			connect(innerView,&Abstract2DInnerView::viewAreaChanged, item,&HelicoItem::refreshItemZoomScale);
		}
	}
	m_datacontrolers.remove(controler);
	return o;
}

QWidget* SeismicSurveyRepOnMap::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new SeismicSurveyPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer * SeismicSurveyRepOnMap::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)
{
	if (m_layer == nullptr)
		m_layer = new SeismicSurveyLayer(this,scene,defaultZDepth,parent);

	return m_layer;
}

AbstractGraphicRep::TypeRep SeismicSurveyRepOnMap::getTypeGraphicRep() {
	return AbstractGraphicRep::Courbe;
}


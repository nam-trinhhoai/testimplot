#include "wellheadlayeronmap.h"
#include "wellheadreponmap.h"
#include "qglimagegriditem.h"
#include "wellhead.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QtSvgWidgets/qgraphicssvgitem.h>
#include <QEvent>
#include <QScreen>
#include <cmath>

WellHeadLayerOnMap::WellHeadLayerOnMap(WellHeadRepOnMap *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent) :
GraphicLayer(scene, defaultZDepth) {
	m_rep=rep;
	m_view = nullptr;

	WellHead* wellHead = dynamic_cast<WellHead*>(rep->data());

	m_item = new QGraphicsSvgItem(":/slicer/icons/45_OILSHOW.svg");
	//scene->addItem(m_item);
	m_item->setZValue(defaultZDepth);
	m_item->setScale(20);
	QRectF rect = m_item->boundingRect();
	m_item->setPos(QPointF(wellHead->x() - rect.width()*10, wellHead->y() - rect.height()*10));
}

WellHeadLayerOnMap::~WellHeadLayerOnMap() {
}
void WellHeadLayerOnMap::show()
{
	m_scene->addItem(m_item);

	QList<QGraphicsView*> views = m_scene->views();
	if (views.size()<1) {
		m_view = nullptr;
		return;
	}
	m_view = views[0];

	m_scaleInit = false;
	updateFromZoom();

	//m_view->installEventFilter(this);
	//connect(m_scene, &QGraphicsScene::sceneRectChanged, this, &WellHeadLayerOnMap::updateFromZoom);
}

void WellHeadLayerOnMap::hide()
{
	//disconnect(m_scene, &QGraphicsScene::sceneRectChanged, this, &WellHeadLayerOnMap::updateFromZoom);
	if (m_view!=nullptr) {
		m_view->removeEventFilter(this);
		m_view = nullptr;
	}

	m_scene->removeItem(m_item);
}

QRectF WellHeadLayerOnMap::boundingRect() const
{
	WellHead* wellHead = dynamic_cast<WellHead*>(m_rep->data());
	QRectF rect = m_item->boundingRect();
	return QRectF(wellHead->x()-rect.width()/2, wellHead->y()-rect.height()/2, rect.width(), rect.height());
}


void WellHeadLayerOnMap::refresh() {
//	m_item->update();
	m_item->update();
}

bool WellHeadLayerOnMap::eventFilter(QObject* watched, QEvent* ev) {
	if (ev->type() == QEvent::Wheel) {
		updateFromZoom();
	}
	return false;
}

void WellHeadLayerOnMap::updateFromZoom() {
	if (m_view==nullptr) {
		return;
	}

	QGraphicsView* view = m_view;

	QScreen* screen = view->screen();
	float inchSize = 0.3f;
	QPoint topLeft(0, 0);
	QPoint topRight(screen->logicalDotsPerInchX(), 0);
	QPoint bottomLeft(0, screen->logicalDotsPerInchY());

//	QSize viewSize = view->viewport()->size();

//	QPoint topLeft(0, 0), topRight(viewSize.width(), 0), bottomLeft(0, viewSize.height());
	QPointF topLeft1 = view->mapToScene(view->mapFromGlobal(topLeft));
	QPointF topRight1 = view->mapToScene(view->mapFromGlobal(topRight));
	QPointF bottomLeft1 = view->mapToScene(view->mapFromGlobal(bottomLeft));

	QPointF dWidthPt = (topRight1 - topLeft1);
	double dWidth = std::sqrt(std::pow(dWidthPt.x(),2) + std::pow(dWidthPt.y(), 2));
	QPointF dHeightPt = (bottomLeft1 - topLeft1);
	double dHeight = std::sqrt(std::pow(dHeightPt.x(),2) + std::pow(dHeightPt.y(), 2));

	QRectF rect = m_item->boundingRect();
	double svgScale = std::min(dWidth / rect.width(), dHeight / rect.height()) * inchSize;

	if (!m_scaleInit || std::fabs(svgScale-m_cachedScale)>std::min(0.001, svgScale/100.0)) {
		WellHead* wellHead = dynamic_cast<WellHead*>(m_rep->data());
		m_item->setScale(svgScale);
		m_item->setPos(QPointF(wellHead->x() - rect.width()*svgScale/2, wellHead->y() - rect.height()*svgScale/2));
	}
	m_cachedScale = svgScale;
	m_scaleInit = true;

//	refresh();
//	m_view->update();
}

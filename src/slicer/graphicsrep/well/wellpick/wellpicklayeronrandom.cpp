#include "wellpicklayeronrandom.h"
#include "wellpickreponrandom.h"
#include "wellpick.h"
#include "marker.h"
#include "wellhead.h"
#include "qglimagegriditem.h"
#include "geometry2dtoolbox.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QGraphicsLineItem>
#include <QDebug>
#include <cmath>

WellPickLayerOnRandom::WellPickLayerOnRandom(WellPickRepOnRandom *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent) :
GraphicLayer(scene, defaultZDepth) {
	m_rep=rep;
	m_parent = parent;

//	WellPick* wellPick = dynamic_cast<WellPick*>(rep->data());

	//m_item = new QGraphicsSvgItem(":/slicer/icons/derick.svg", parent);
	//scene->addItem(m_item);
	//m_item->setZValue(defaultZDepth);
	//m_item->setScale(20);

//	m_lineItem = new QGraphicsLineItem(0, 0, 1, 1, parent);
//	m_lineItem->setZValue(defaultZDepth);
//	QPen pen = m_lineItem->pen();
//	if (wellPick->currentMarker()!=nullptr) {
//		pen.setColor(wellPick->currentMarker()->color());
//	}
//	pen.setCosmetic(true);
//	pen.setWidth(5);
//	m_lineItem->setPen(pen);

//	QRectF rect = m_item->boundingRect();
//	m_item->setPos(QPointF(wellHead->x() - rect.width()*10, wellHead->y() - rect.height()*10));
}

WellPickLayerOnRandom::~WellPickLayerOnRandom() {
}

QGraphicsLineItem* WellPickLayerOnRandom::getLineItem() const {
	QGraphicsLineItem* lineItem;
	WellPick* wellPick = dynamic_cast<WellPick*>(m_rep->data());
	lineItem = new QGraphicsLineItem(0, 0, 1, 1, m_parent);
	lineItem->setZValue(m_defaultZDepth);
	QPen pen = lineItem->pen();
	if (wellPick->currentMarker()!=nullptr) {
		pen.setColor(wellPick->currentMarker()->color());
	}
	pen.setCosmetic(true);
	pen.setWidth(5);
	lineItem->setPen(pen);
	return lineItem;
}

void WellPickLayerOnRandom::show() {
	if (m_requestedShown) {
		return;
	}
	internalShow();
	m_requestedShown = true;
}

void WellPickLayerOnRandom::hide() {
	if (!m_requestedShown) {
		return;
	}
	internalHide();
	m_requestedShown = false;
}

void WellPickLayerOnRandom::internalShow()
{
	bool isValid;
	QList<QPointF> points = m_rep->getCurrentPointList(&isValid);

	if (isValid) {
		m_rep->searchWellBoreRep();
		isValid = m_rep->isLinkedRepValid() && m_rep->linkedRepShown();
	}

	if (isValid) {
		for (QPointF point : points) {
			QGraphicsLineItem* lineItem = getLineItem();
			lineItem->setLine(point.x()-0.5, point.y()-0.5, point.x()+0.5, point.y()+0.5);
			lineItem->setToolTip(m_rep->wellPick()->markerName());
			m_scene->addItem(lineItem);
			m_lineItems.push_back(lineItem);
		}

		m_isShown = true;
	}
	//connect(m_scene, &QGraphicsScene::sceneRectChanged, this, &WellHeadLayerOnMap::updateFromZoom);
}

void WellPickLayerOnRandom::internalHide()
{
	//disconnect(m_scene, &QGraphicsScene::sceneRectChanged, this, &WellHeadLayerOnMap::updateFromZoom);

	//m_scene->removeItem(m_item);
	for (std::size_t i=0; i<m_lineItems.size(); i++) {
		m_scene->removeItem(m_lineItems[i]);
		delete m_lineItems[i];
	}
	m_lineItems.clear();

	m_isShown = false;
}

QRectF WellPickLayerOnRandom::boundingRect() const
{
	QRectF rect;

	bool isValid;
	QList<QPointF> points = m_rep->getCurrentPointList(&isValid);

	if (isValid) {
		double axis;
		bool firstIf = true;
		for (QPointF point : points) {
			QRectF currentRect(point.x()-0.5, point.y()-0.5, 1, 1);
			if (firstIf) {
				firstIf = false;
				rect = currentRect;
			} else {
				rect = rect.united(currentRect);
			}
		}

	}
	return rect;
}

void WellPickLayerOnRandom::refresh() {
	if (m_requestedShown) {
		if (m_isShown) {
			internalHide();
		}
		internalShow();
		for (QGraphicsLineItem* lineItem : m_lineItems) {
			lineItem->update();
		}
	}
}

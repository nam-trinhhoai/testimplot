#include "wellpicklayeronslice.h"
#include "wellpickreponslice.h"
#include "wellpick.h"
#include "marker.h"
#include "wellhead.h"
#include "qglimagegriditem.h"
#include "geometry2dtoolbox.h"
#include "abstractsectionview.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QGraphicsLineItem>
#include <QDebug>
#include <cmath>

WellPickLayerOnSlice::WellPickLayerOnSlice(WellPickRepOnSlice *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent) :
GraphicLayer(scene, defaultZDepth) {
	m_rep=rep;
	m_view = nullptr;

	WellPick* wellPick = dynamic_cast<WellPick*>(rep->data());

	//m_item = new QGraphicsSvgItem(":/slicer/icons/derick.svg", parent);
	//scene->addItem(m_item);
	//m_item->setZValue(defaultZDepth);
	//m_item->setScale(20);

	m_lineItem = new QGraphicsLineItem(0, 0, 1, 1, parent);
	m_lineItem->setZValue(defaultZDepth);
	QPen pen = m_lineItem->pen();
	if (wellPick->currentMarker()!=nullptr) {
		pen.setColor(wellPick->currentMarker()->color());
	}
	pen.setCosmetic(true);
	pen.setWidth(5);
	m_lineItem->setPen(pen);
	m_lineItem->setToolTip(rep->wellPick()->markerName());

//	QRectF rect = m_item->boundingRect();
//	m_item->setPos(QPointF(wellHead->x() - rect.width()*10, wellHead->y() - rect.height()*10));
}

WellPickLayerOnSlice::~WellPickLayerOnSlice() {
}

void WellPickLayerOnSlice::show()
{
	AbstractSectionView* view = dynamic_cast<AbstractSectionView*>(m_rep->view());
	if (view==nullptr || !view->isMapRelationSet()) {
		return;
	}

	bool isValid;
	QVector3D point = m_rep->getCurrentPoint(&isValid);

	if (isValid) {
		m_rep->searchWellBoreRep();
		isValid = m_rep->isLinkedRepValid() && m_rep->linkedRepShown();
	}
	if (!isValid){
		return;
	}
//	WellHead* wellHead = dynamic_cast<WellHead*>(m_rep->data());

	std::pair<QPointF, QPointF> segmentMap = view->getSectionSegment();
//	double imageAX, imageAY, imageBX, imageBY, imageWellX, imageWellY;
//	view->inlineXLineToXY()->worldToImage(segmentMap.first.x(), segmentMap.first.y(), imageAX, imageAY);
//	view->inlineXLineToXY()->worldToImage(segmentMap.second.x(), segmentMap.second.y(), imageBX, imageBY);

//	imageWellX = point.x();
//	imageWellY = point.y();

	double worldWellX, worldWellY;

	bool ok = view && view->inlineXLineToXY();
	if (ok) {
		view->inlineXLineToXY()->imageToWorld(point.x(), point.y(), worldWellX, worldWellY);
	}

	std::pair<double, QPointF> distanceAndProjection;
	//	std::pair<QPointF, QPointF> segment = std::pair<QPointF, QPointF>(QPointF(imageAX, imageAY), QPointF(imageBX, imageBY));
	if (ok) {
		distanceAndProjection = getPointProjectionOnLine(QPointF(worldWellX, worldWellY), segmentMap, &ok);
	}

//	std::pair<QPointF, QPointF> segment = view->getSectionSegment();
//	std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(QPointF(wellHead->x(),wellHead->y()), segment);

	WellHead* wellHead = dynamic_cast<WellPick*>(m_rep->data())->wellBore()->wellHead();
	if (ok && distanceAndProjection.first<m_rep->displayDistance()) {
		QList<QGraphicsView*> views = m_scene->views();
		if (views.size()<1) {
			m_view = nullptr;
			return;
		}
		m_view = views[0];

//		updateFromZoom();
		double axis;
		if (view->viewType()==ViewType::InlineView) {
			axis = point.x();
		} else {
			axis = point.y();
		}
		m_lineItem->setLine(axis-0.5, point.z()-0.5, axis+0.5, point.z()+0.5);
		m_scene->addItem(m_lineItem);
		//m_scene->addItem(m_item);

		m_view->installEventFilter(this);
		m_isVisibleOnSection = true;
		m_isShown = true;
	}
	//connect(m_scene, &QGraphicsScene::sceneRectChanged, this, &WellHeadLayerOnMap::updateFromZoom);
}

void WellPickLayerOnSlice::hide() {
	hide(false);
}

void WellPickLayerOnSlice::hide(bool soft)
{

	//disconnect(m_scene, &QGraphicsScene::sceneRectChanged, this, &WellHeadLayerOnMap::updateFromZoom);
	if (m_isVisibleOnSection) {
		if (m_view!=nullptr) {
			m_view->removeEventFilter(this);
			m_view = nullptr;
		}

		//m_scene->removeItem(m_item);
		m_scene->removeItem(m_lineItem);
	}

	m_isVisibleOnSection = false;
	if (!soft) {
		m_isShown = false;
	}
}

QRectF WellPickLayerOnSlice::boundingRect() const
{
	QRectF rect;

	bool isValid;
	QVector3D point = m_rep->getCurrentPoint(&isValid);

	if (isValid) {
		double axis;
		AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(m_rep->view());
		if (sectionView->viewType()==ViewType::InlineView) {
			axis = point.x();
		} else {
			axis = point.y();
		}
		rect = QRectF(axis-0.5, point.z()-0.5, 1, 1);
	}
	return rect;
}

void WellPickLayerOnSlice::reloadItems() {
	if (m_isShown) {
		bool isValid;
		QVector3D point = m_rep->getCurrentPoint(&isValid);

		AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(m_rep->view());
		isValid = isValid && sectionView!=nullptr && sectionView->isMapRelationSet();

		if (isValid) {
			WellHead* wellHead = dynamic_cast<WellPick*>(m_rep->data())->wellBore()->wellHead();

			std::pair<QPointF, QPointF> segmentMap = sectionView->getSectionSegment();
//			double imageAX, imageAY, imageBX, imageBY, imageWellX, imageWellY;
//			sectionView->inlineXLineToXY()->worldToImage(segmentMap.first.x(), segmentMap.first.y(), imageAX, imageAY);
//			sectionView->inlineXLineToXY()->worldToImage(segmentMap.second.x(), segmentMap.second.y(), imageBX, imageBY);
//
//			imageWellX = point.x();
//			imageWellY = point.y();

//			std::pair<QPointF, QPointF> segment = std::pair<QPointF, QPointF>(QPointF(imageAX, imageAY), QPointF(imageBX, imageBY));
			double worldWellX, worldWellY;
			sectionView->inlineXLineToXY()->imageToWorld(point.x(), point.y(), worldWellX, worldWellY);

			std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(QPointF(worldWellX, worldWellY), segmentMap, &isValid);
			isValid = isValid && distanceAndProjection.first<m_rep->displayDistance();

			if (isValid) {
				m_rep->searchWellBoreRep();
				isValid = m_rep->isLinkedRepValid() && m_rep->linkedRepShown();
			}
		}
		if (!isValid) {
			hide(true);
		} else if (!m_isVisibleOnSection) {
			show();
		} else {
			hide(true);
			show();
		}
	} else {
		show();
	}
	refresh();
}

void WellPickLayerOnSlice::refresh() {
	m_lineItem->update();
}

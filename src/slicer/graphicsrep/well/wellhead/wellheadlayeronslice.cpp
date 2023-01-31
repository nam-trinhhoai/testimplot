#include "wellheadlayeronslice.h"
#include "wellheadreponslice.h"
#include "qglimagegriditem.h"
#include "wellhead.h"
#include "geometry2dtoolbox.h"
#include "abstractsectionview.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "slicerep.h"
#include "seismic3dabstractdataset.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QGraphicsLineItem>
#include <qgraphicssvgitem.h>
#include <QEvent>
#include <QDebug>
#include <cmath>

WellHeadLayerOnSlice::WellHeadLayerOnSlice(WellHeadRepOnSlice *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent) :
GraphicLayer(scene, defaultZDepth) {
	m_rep=rep;
	m_view = nullptr;

	WellHead* wellHead = dynamic_cast<WellHead*>(rep->data());

	//m_item = new QGraphicsSvgItem(":/slicer/icons/derick.svg", parent);
	//scene->addItem(m_item);
	//m_item->setZValue(defaultZDepth);
	//m_item->setScale(20);

	m_lineItem = new QGraphicsLineItem(0, 0, 0, 0, parent);
	m_lineItem->setZValue(defaultZDepth);
//	QRectF rect = m_item->boundingRect();
//	m_item->setPos(QPointF(wellHead->x() - rect.width()*10, wellHead->y() - rect.height()*10));
}

WellHeadLayerOnSlice::~WellHeadLayerOnSlice() {
}
void WellHeadLayerOnSlice::show()
{
	AbstractSectionView* view = dynamic_cast<AbstractSectionView*>(m_rep->view());
	if (view==nullptr || !view->isMapRelationSet()) {
		return;
	}

	WellHead* wellHead = dynamic_cast<WellHead*>(m_rep->data());

	std::pair<QPointF, QPointF> segmentMap = view->getSectionSegment();
	double imageAX, imageAY, imageBX, imageBY, imageWellX, imageWellY;
	view->inlineXLineToXY()->worldToImage(segmentMap.first.x(), segmentMap.first.y(), imageAX, imageAY);
	view->inlineXLineToXY()->worldToImage(segmentMap.second.x(), segmentMap.second.y(), imageBX, imageBY);
	view->inlineXLineToXY()->worldToImage(wellHead->x(), wellHead->y(), imageWellX, imageWellY);

	bool ok;
	std::pair<QPointF, QPointF> segment = std::pair<QPointF, QPointF>(QPointF(imageAX, imageAY), QPointF(imageBX, imageBY));
	std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(QPointF(imageWellX,imageWellY), segment, &ok);

//	std::pair<QPointF, QPointF> segment = view->getSectionSegment();
//	std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(QPointF(wellHead->x(),wellHead->y()), segment);

	if (distanceAndProjection.first<m_rep->displayDistance()) {
		QList<QGraphicsView*> views = m_scene->views();
		if (views.size()<1) {
			m_view = nullptr;
			return;
		}
		m_view = views[0];

		updateFromZoom();
		m_scene->addItem(m_lineItem);
		//m_scene->addItem(m_item);

		m_view->installEventFilter(this);
		m_isVisibleOnSection = true;
		m_isShown = true;
	}
	//connect(m_scene, &QGraphicsScene::sceneRectChanged, this, &WellHeadLayerOnMap::updateFromZoom);
}

void WellHeadLayerOnSlice::hide() {
	hide(false);
}

void WellHeadLayerOnSlice::hide(bool soft)
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

QRectF WellHeadLayerOnSlice::boundingRect() const
{
 	AbstractSectionView* originView = dynamic_cast<AbstractSectionView*>(m_rep->view());

	if (originView==nullptr || !originView->isMapRelationSet() || originView->firstSlice()==nullptr) {
 		return QRectF();
	}

	WellHead* wellHead = dynamic_cast<WellHead*>(m_rep->data());
	SliceRep* slice = originView->firstSlice();
	Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(slice->data());


	bool ok;
	std::pair<QPointF, QPointF> segment = originView->getSectionSegment();
	std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(QPointF(wellHead->x(),wellHead->y()), segment, &ok);


	double imageX, imageY, sampleMin, sampleMax;
	originView->inlineXLineToXY()->worldToImage(distanceAndProjection.second.x(), distanceAndProjection.second.y(), imageX, imageY);
	dataset->sampleTransformation()->direct(0, sampleMin);
	dataset->sampleTransformation()->direct(dataset->depth()-1, sampleMax);
	QPointF posWell, bottomWell;
	if (originView->viewType()==ViewType::InlineView) {
		posWell = QPointF(imageX, sampleMin);
		bottomWell = QPointF(imageX, sampleMax);
	} else {
		posWell = QPointF(imageY, sampleMin);
		bottomWell = QPointF(imageY, sampleMax);
	}

	//QRectF rect = m_item->boundingRect();
	//QRectF iconRect = QRectF(posWell.x()-rect.width()/2, posWell.y()-rect.height()/2, rect.width(), rect.height());
	return /*iconRect.united*/(QRectF(posWell, bottomWell));
}

void WellHeadLayerOnSlice::reloadItems() {
	if (m_isShown) {
		AbstractSectionView* sectionView = dynamic_cast<AbstractSectionView*>(m_rep->view());
		bool isValid = sectionView!=nullptr && sectionView->isMapRelationSet();

		if (isValid) {
			WellHead* wellHead = dynamic_cast<WellHead*>(m_rep->data());

			std::pair<QPointF, QPointF> segmentMap = sectionView->getSectionSegment();
			double imageAX, imageAY, imageBX, imageBY, imageWellX, imageWellY;
			sectionView->inlineXLineToXY()->worldToImage(segmentMap.first.x(), segmentMap.first.y(), imageAX, imageAY);
			sectionView->inlineXLineToXY()->worldToImage(segmentMap.second.x(), segmentMap.second.y(), imageBX, imageBY);
			sectionView->inlineXLineToXY()->worldToImage(wellHead->x(), wellHead->y(), imageWellX, imageWellY);

			std::pair<QPointF, QPointF> segment = std::pair<QPointF, QPointF>(QPointF(imageAX, imageAY), QPointF(imageBX, imageBY));
			std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(QPointF(imageWellX,imageWellY), segment, &isValid);
			isValid = isValid && distanceAndProjection.first<m_rep->displayDistance();
		}
		if (!isValid) {
			hide(true);
		} else if (!m_isVisibleOnSection) {
			show();
		}
	} else {
		show();
	}
	refresh();
}

void WellHeadLayerOnSlice::refresh() {
	//m_item->update();
	m_lineItem->update();
}

bool WellHeadLayerOnSlice::eventFilter(QObject* watched, QEvent* ev) {
	if (ev->type() == QEvent::Wheel) {
		updateFromZoom();
	}
	return false;
}

void WellHeadLayerOnSlice::updateFromZoom() {
	if (m_view==nullptr) {
		return;
	}

 	AbstractSectionView* originView = dynamic_cast<AbstractSectionView*>(m_rep->view());
	if (originView==nullptr || !originView->isMapRelationSet() || originView->firstSlice()==nullptr) {
 		return;
	}

	WellHead* wellHead = dynamic_cast<WellHead*>(m_rep->data());
	SliceRep* slice = originView->firstSlice();
	Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(slice->data());

	bool ok;
	std::pair<QPointF, QPointF> segment = originView->getSectionSegment();
	std::pair<double, QPointF> distanceAndProjection = getPointProjectionOnLine(QPointF(wellHead->x(),wellHead->y()), segment, &ok);


	double imageX, imageY, sampleMin, sampleMax;
	originView->inlineXLineToXY()->worldToImage(distanceAndProjection.second.x(), distanceAndProjection.second.y(), imageX, imageY);
	dataset->sampleTransformation()->direct(0, sampleMin);
	dataset->sampleTransformation()->direct(dataset->height()-1, sampleMax);
	QPointF posWell;
	if (originView->viewType()==ViewType::InlineView) {
		posWell = QPointF(imageX, sampleMin);
	} else {
		posWell = QPointF(imageY, sampleMin);
	}

	QGraphicsView* view = m_view;

	QSize viewSize = view->viewport()->size();

	QPoint topLeft(0, 0), topRight(viewSize.width(), 0), bottomLeft(0, viewSize.height());
	QPointF topLeft1 = view->mapToScene(topLeft);
	QPointF topRight1 = view->mapToScene(topRight);
	QPointF bottomLeft1 = view->mapToScene(bottomLeft);

	QPointF dWidthPt = (topRight1 - topLeft1);
	double dWidth = std::sqrt(std::pow(dWidthPt.x(),2) + std::pow(dWidthPt.y(), 2));
	QPointF dHeightPt = (bottomLeft1 - topLeft1);
	double dHeight = std::sqrt(std::pow(dHeightPt.x(),2) + std::pow(dHeightPt.y(), 2));

	//QRectF rect = m_item->boundingRect();
	//double svgScale = std::min(dWidth / rect.width(), dHeight / rect.height()) / 10;

	//m_item->setScale(svgScale);
	//m_item->setPos(QPointF(posWell.x() - rect.width()*svgScale/2, posWell.y() - rect.height()*svgScale/2));

	m_lineItem->setLine(posWell.x(), posWell.y(), posWell.x(), sampleMax);

	double penScale = std::min(dWidth, dHeight) / 50;

	QPen pen = m_lineItem->pen();
	pen.setWidthF(penScale/5);
	m_lineItem->setPen(pen);
}

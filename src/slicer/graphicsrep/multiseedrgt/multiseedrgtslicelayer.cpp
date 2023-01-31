#include "multiseedrgtslicelayer.h"
#include "multiseedslicerep.h"
#include "multiseedhorizon.h"
#include "seismic3ddataset.h"
#include "slicerep.h"
#include "affine2dtransformation.h"

#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <cmath>

MultiSeedRgtSliceLayer::MultiSeedRgtSliceLayer(MultiSeedRgtSliceRep* rep, QGraphicsScene *scene,
		int defaultZDepth, QGraphicsItem* parent) : GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_curveMain.reset(new Curve(m_scene, parent));
	m_curveTop.reset(new Curve(m_scene, parent));
	m_curveBottom.reset(new Curve(m_scene, parent));

	m_curveReference.resize(100);
	for (int i=0; i<m_curveReference.size(); i++)
	{
		m_curveReference[i].reset(new Curve(m_scene, parent));
	}

	m_curveMain->setZValue(m_defaultZDepth);
	m_curveTop->setZValue(m_defaultZDepth);
	m_curveBottom->setZValue(m_defaultZDepth);


	refresh();
	show();
}

MultiSeedRgtSliceLayer::~MultiSeedRgtSliceLayer() {}

void MultiSeedRgtSliceLayer::show() {
	/*
	if(isShown) {
		hide(); // hide then show to allow update with different horizon mode
	}
	isShown = true;
	if (m_rep->getData()->getHorizonMode()==HORIZONMODE::DEFAULT) {
		m_curveMain->addToScene();
		m_curveTop->addToScene();
		m_curveBottom->addToScene();
		m_curveTop->hide();
		m_curveBottom->hide();
		for (int i=0; i<m_curveReference.size(); i++)
		{
			m_curveReference[i]->addToScene();
		}
	} else {
		m_curveMain->addToScene();
		m_curveTop->addToScene();
		m_curveBottom->addToScene();
		for (int i=0; i<m_curveReference.size(); i++)
		{
			m_curveReference[i]->addToScene();
		}
	}
	for (QGraphicsEllipseItem* e: m_seedsRepresentation) {
		m_scene->addItem(e);
	}
	*/
}

void MultiSeedRgtSliceLayer::hide() {
	/*
	if (!isShown) {
		return; // already hidden
	}
	isShown = false;

	m_curveMain->removeFromScene();
	m_curveTop->removeFromScene();
	m_curveBottom->removeFromScene();
	for (int i=0; i<m_curveReference.size(); i++)
	{
		m_curveReference[i]->removeFromScene();
	}
	for (QGraphicsEllipseItem* e : m_seedsRepresentation) {
		m_scene->removeItem(e);
	}
	*/
}

QRectF MultiSeedRgtSliceLayer::boundingRect() const {
//	QRectF rect = m_curveTop->boundingRect();
//	rect.united(m_curveBottom->boundingRect());

	// copied from CUDAImageBuffer worldExtent to give the same result
	/*
	double width;
	if (m_rep->direction()==SliceDirection::Inline) {
		width = m_rep->getData()->seismic()->width();
	} else {
		width = m_rep->getData()->seismic()->depth();
	}

	double ij[8] = { 0.0, (double) m_rep->getData()->getDeltaTop(), (double) width, (double) m_rep->getData()->getDeltaTop(),
			0.0, (double) m_rep->getData()->seismic()->height() + m_rep->getData()->getDeltaBottom(),
			(double) width, (double) m_rep->getData()->seismic()->height() + m_rep->getData()->getDeltaBottom() };

	double xMin = std::numeric_limits<double>::max();
	double yMin = std::numeric_limits<double>::max();

	double xMax = std::numeric_limits<double>::min();
	double yMax = std::numeric_limits<double>::min();
	double x, y;
	for (int i = 0; i < 4; i++) {
		x = ij[2 * i];
		y = ij[2 * i + 1];

		if (m_rep->direction()==SliceDirection::Inline) {
			m_rep->getData()->seismic()->ijToInlineXlineTransfoForInline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
		} else {
			m_rep->getData()->seismic()->ijToInlineXlineTransfoForXline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
		}

		xMin = std::min(xMin, x);
		yMin = std::min(yMin, y);

		xMax = std::max(xMax, x);
		yMax = std::max(yMax, y);
	}
	QRectF rect(xMin, yMin, xMax - xMin, yMax - yMin);
	return rect;
	*/
	return m_curveTop->boundingRect();
}

void MultiSeedRgtSliceLayer::refreshPolygons() {
	/*
	m_curveMain->setPolygon(m_rep->getMainPolygon());
	m_curveTop->setPolygon(m_rep->getTopPolygon());
	m_curveBottom->setPolygon(m_rep->getBottomPolygon());
	m_curveMain->setPen(m_rep->getPen());
	m_curveTop->setPen(m_rep->getPenDelta());
	m_curveBottom->setPen(m_rep->getPenDelta());

	// apply transform
	QTransform mainTransform;
	if (m_rep->direction()==SliceDirection::Inline) {
		mainTransform = QTransform(dynamic_cast<MultiSeedHorizon*>(m_rep->data())->seismic()->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
	} else {
		mainTransform = QTransform(dynamic_cast<MultiSeedHorizon*>(m_rep->data())->seismic()->ijToInlineXlineTransfoForXline()->imageToWorldTransformation().toTransform());
	}
	m_curveMain->setTransform(mainTransform);
	m_curveTop->setTransform(mainTransform);
	m_curveBottom->setTransform(mainTransform);
//	m_curveMain->setZValue(m_defaultZDepth);
//	m_curveTop->setZValue(m_defaultZDepth);
//	m_curveBottom->setZValue(m_defaultZDepth);


	std::vector<std::vector<QPolygon>> poly = m_rep->getReferencePolygon();
	QPen pen0(Qt::blue, 3);
	for (int i=0; i<poly.size(); i++)
	{
		QList<QPolygon> tmpList(poly[i].begin(), poly[i].end());
		m_curveReference[i]->setPolygons(tmpList);
		// m_curveReference[i]->setPen(m_rep->getPen());
		m_curveReference[i]->setPen(pen0);
		m_curveReference[i]->setTransform(mainTransform);
		m_curveReference[i]->setZValue(m_defaultZDepth);
	}
	QPolygon emptyPoly;
	for (int i=poly.size(); i<m_curveReference.size(); i++) {
		m_curveReference[i]->setPolygon(emptyPoly);
	}

	if (m_rep->getData()->getHorizonMode()==HORIZONMODE::DEFAULT) {
		m_curveBottom->hide();
		m_curveTop->hide();
	} else {
		m_curveBottom->show();
		m_curveTop->show();
	}
	*/

}

void MultiSeedRgtSliceLayer::refresh() {
	/*
	refreshPolygons();

	for (QGraphicsEllipseItem* item : m_seedsRepresentation) {
		if (isShown) {
			m_scene->removeItem(item);
		}
		delete item;
	}
	m_seedsRepresentation.clear();

	for (const std::pair<std::size_t, RgtSeed>& seed : m_rep->getData()->getMap()) {
		bool isPointInSection = (m_rep->direction()==SliceDirection::Inline && seed.second.z==m_rep->currentSliceIJPosition()) ||
								(m_rep->direction()==SliceDirection::XLine && seed.second.y==m_rep->currentSliceIJPosition());
		if (isPointInSection) {
			float y = seed.second.x;
			float x;
			float w = 1;
			float h = 1;
			if (m_rep->direction()==SliceDirection::Inline) {
				x = seed.second.y;
			} else {
				x = seed.second.z;
			}
			x-=w/2;
			y-=h/2;


			QGraphicsEllipseItem* item = new QGraphicsEllipseItem(x, y, w, h);
			QPen penPoints = m_rep->getPenPoints();
			penPoints.setCosmetic(true);
			penPoints.setWidth(2);
			item->setPen(penPoints);
			if (isShown) {
				m_scene->addItem(item);
			}
			m_seedsRepresentation.push_back(item);
//			m_seedsRepresentation.push_back(m_scene->addEllipse(x, y, w, h, m_rep->getPenPoints()));
			QTransform mainTransform;
			if (m_rep->direction()==SliceDirection::Inline) {
				mainTransform = QTransform(dynamic_cast<MultiSeedHorizon*>(m_rep->data())->seismic()->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
			} else {
				mainTransform = QTransform(dynamic_cast<MultiSeedHorizon*>(m_rep->data())->seismic()->ijToInlineXlineTransfoForXline()->imageToWorldTransformation().toTransform());
			}

			m_seedsRepresentation.back()->setTransform(mainTransform);
			m_seedsRepresentation.back()->setZValue(m_defaultZDepth);
		}
	}



	if (isShown) {
		//show();
	} else {
		hide();
	}
	*/
}

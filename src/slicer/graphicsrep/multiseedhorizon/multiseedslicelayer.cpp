#include "multiseedslicelayer.h"
#include "multiseedslicerep.h"
#include "multiseedhorizon.h"
#include "fixedlayerfromdataset.h"
#include "seismic3ddataset.h"
#include "slicerep.h"
#include "affine2dtransformation.h"

#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <cmath>

MultiSeedSliceLayer::MultiSeedSliceLayer(MultiSeedSliceRep* rep, QGraphicsScene *scene,
		int defaultZDepth, QGraphicsItem* parent) : GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_parent = parent;
	m_curveMain.reset(new Curve(m_scene, parent));
	m_curveTop.reset(new Curve(m_scene, parent));
	m_curveBottom.reset(new Curve(m_scene, parent));

	std::vector<std::shared_ptr<FixedLayerFromDataset>> references = m_rep->getData()->getReferences();
	m_curveReference.resize(references.size());
	for (int i=0; i<m_curveReference.size(); i++)
	{
		m_curveReference[i].reset(new ReferenceCurveWrapper(references[i].get()));
		m_curveReference[i]->setCurve(new Curve(m_scene, parent));
	}

	m_curveMain->setZValue(m_defaultZDepth);
	m_curveTop->setZValue(m_defaultZDepth);
	m_curveBottom->setZValue(m_defaultZDepth);


	refresh();
	show();

	connect(m_rep->getData(), &MultiSeedHorizon::referencesChanged, this, &MultiSeedSliceLayer::referencesChanged);
}

MultiSeedSliceLayer::~MultiSeedSliceLayer() {
	disconnect(m_rep->getData(), &MultiSeedHorizon::referencesChanged, this, &MultiSeedSliceLayer::referencesChanged);
}

void MultiSeedSliceLayer::show() {
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
			m_curveReference[i]->curve()->addToScene();
		}
	} else {
		m_curveMain->addToScene();
		m_curveTop->addToScene();
		m_curveBottom->addToScene();
		for (int i=0; i<m_curveReference.size(); i++)
		{
			m_curveReference[i]->curve()->addToScene();
		}
	}
	for (QGraphicsEllipseItem* e: m_seedsRepresentation) {
		m_scene->addItem(e);
	}
}

void MultiSeedSliceLayer::hide() {
	if (!isShown) {
		return; // already hidden
	}
	isShown = false;

	m_curveMain->removeFromScene();
	m_curveTop->removeFromScene();
	m_curveBottom->removeFromScene();
	for (int i=0; i<m_curveReference.size(); i++)
	{
		m_curveReference[i]->curve()->removeFromScene();
	}
	for (QGraphicsEllipseItem* e : m_seedsRepresentation) {
		m_scene->removeItem(e);
	}
}

QRectF MultiSeedSliceLayer::boundingRect() const {
//	QRectF rect = m_curveTop->boundingRect();
//	rect.united(m_curveBottom->boundingRect());

	// copied from CUDAImageBuffer worldExtent to give the same result
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
}

void MultiSeedSliceLayer::refreshPolygons() {
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
	for (int i=0; i<std::min(m_curveReference.size(), poly.size()); i++)
	{
		QList<QPolygon> tmpList(poly[i].begin(), poly[i].end());
		m_curveReference[i]->curve()->setPolygons(tmpList);
		// m_curveReference[i]->setPen(m_rep->getPen());
		QPen pen = m_curveReference[i]->curve()->getPen();
		pen.setWidth(pen0.width());
		m_curveReference[i]->curve()->setPen(pen);
		m_curveReference[i]->curve()->setTransform(mainTransform);
		m_curveReference[i]->curve()->setZValue(m_defaultZDepth);
	}
	QPolygon emptyPoly;
	for (int i=poly.size(); i<m_curveReference.size(); i++) {
		m_curveReference[i]->curve()->setPolygon(emptyPoly);
	}

	if (m_rep->getData()->getHorizonMode()==HORIZONMODE::DEFAULT) {
		m_curveBottom->hide();
		m_curveTop->hide();
	} else {
		m_curveBottom->show();
		m_curveTop->show();
	}

}

void MultiSeedSliceLayer::refresh() {
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

			/*x *= m_pixelSampleRateX;
			y *= m_pixelSampleRateY;
			w *= m_pixelSampleRateX;
			h *= m_pixelSampleRateY;*/
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


	// jd
	/*
	std::vector<std::vector<RgtSeed>> seeds = m_rep->getData()->getMap2();

	for (int n=0; n<seeds.size(); n++)
	{
		std::vector<RgtSeed> seed0 = seeds[n];
		QPen pen0(Qt::blue, 2);
		for (int k=0; k<seed0.size(); k++)
		{
			RgtSeed seed = seed0[k];
			bool isPointInSection = (m_rep->direction()==SliceDirection::Inline && seed.z==m_rep->currentSliceIJPosition()) ||
												(m_rep->direction()==SliceDirection::XLine && seed.y==m_rep->currentSliceIJPosition());
			if (isPointInSection) {
				float y = seed.x;
				float x;
				float w = 1;
				float h = 1;
				if (m_rep->direction()==SliceDirection::Inline) {
					x = seed.y;
				} else {
					x = seed.z;
				}
				x-=w/2;
				y-=h/2;

				m_seedsRepresentation.push_back(m_scene->addEllipse(x, y, w, h, pen0)); //m_rep->getPenPoints()));
				QTransform mainTransform;
				if (m_rep->direction()==SliceDirection::Inline) {
					mainTransform = QTransform(dynamic_cast<Seismic3DDataset*>(dynamic_cast<MultiSeedHorizon*>(m_rep->data())->seismic()->data())->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
				} else {
					mainTransform = QTransform(dynamic_cast<Seismic3DDataset*>(dynamic_cast<MultiSeedHorizon*>(m_rep->data())->seismic()->data())->ijToInlineXlineTransfoForXline()->imageToWorldTransformation().toTransform());
				}

				m_seedsRepresentation.back()->setTransform(mainTransform);
				m_seedsRepresentation.back()->setZValue(m_defaultZDepth);
			}
		}
	}
	*/



	if (isShown) {
		//show();
	} else {
		hide();
	}
}

void MultiSeedSliceLayer::referencesChanged() {
	std::vector<std::shared_ptr<FixedLayerFromDataset>> references = m_rep->getData()->getReferences();
	std::map<int, std::shared_ptr<FixedLayerFromDataset>> newSelection;
	std::vector<bool> isItemInSelection;
	isItemInSelection.resize(m_curveReference.size(), false);

	for (int i=0; i<references.size(); i++) {
		std::size_t indexLoaded = 0;
		while(indexLoaded<m_curveReference.size() && references[i]->name().compare(m_curveReference[indexLoaded]->layer()->name())) {
			indexLoaded++;
		}
		if (indexLoaded==m_curveReference.size()) {
			newSelection[i] = references[i];
		} else {
			isItemInSelection[indexLoaded] = true;
		}
	}

	for(long indexLoaded=m_curveReference.size()-1; indexLoaded>=0; indexLoaded--) {
		if (!isItemInSelection[indexLoaded]) {
			m_curveReference.erase(m_curveReference.begin()+indexLoaded);
		}
	}

	for (auto it=newSelection.begin(); it!=newSelection.end(); it++) {
		std::shared_ptr<ReferenceCurveWrapper> wrapper = std::make_shared<ReferenceCurveWrapper>(it->second.get());
		wrapper->setCurve(new Curve(m_scene, m_parent));

		m_curveReference.insert(m_curveReference.begin()+it->first, wrapper);

		if (isShown) {
			m_curveReference[it->first]->curve()->addToScene();
		} else {
			m_curveReference[it->first]->curve()->removeFromScene();
		}
	}

	refreshPolygons();
}

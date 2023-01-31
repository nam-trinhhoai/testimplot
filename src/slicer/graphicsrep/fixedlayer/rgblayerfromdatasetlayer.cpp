#include "rgblayerfromdatasetlayer.h"

#include <QGraphicsScene>
#include "rgbqglcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "rgblayerfromdatasetrep.h"
#include "rgblayerfromdataset.h"

RgbLayerFromDatasetLayer::RgbLayerFromDatasetLayer(RgbLayerFromDatasetRep *rep, QGraphicsScene *scene,
		int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_item = new RGBQGLCUDAImageItem(rep->isoSurfaceHolder(),
			rep->image(), 0, parent);
	m_item->setZValue(defaultZDepth);
	connect(m_rep->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->image(), SIGNAL(rangeChanged(unsigned int , const QVector2D &)), this,
			SLOT(refresh()));
	connect(m_rep->image()->get(0), SIGNAL(lookupTableChanged(const LookupTable &)), this,
				SLOT(refresh()));
	connect(m_rep->image()->get(1), SIGNAL(lookupTableChanged(const LookupTable &)), this,
				SLOT(refresh()));
	connect(m_rep->image()->get(2), SIGNAL(lookupTableChanged(const LookupTable &)), this,
				SLOT(refresh()));
}

RgbLayerFromDatasetLayer::~RgbLayerFromDatasetLayer() {
}

void RgbLayerFromDatasetLayer::show() {
	m_scene->addItem(m_item);
}
void RgbLayerFromDatasetLayer::hide() {
	m_scene->removeItem(m_item);
}

QRectF RgbLayerFromDatasetLayer::boundingRect() const {
	return m_rep->image()->get(0)->worldExtent();
}

void RgbLayerFromDatasetLayer::refresh() {
	m_item->update();
}


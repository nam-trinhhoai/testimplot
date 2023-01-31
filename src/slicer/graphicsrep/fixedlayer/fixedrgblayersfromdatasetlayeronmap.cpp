#include "fixedrgblayersfromdatasetlayeronmap.h"
#include <QGraphicsScene>
#include "rgbqglcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "fixedrgblayersfromdatasetrep.h"
#include "fixedrgblayersfromdataset.h"

FixedRGBLayersFromDatasetLayerOnMap::FixedRGBLayersFromDatasetLayerOnMap(FixedRGBLayersFromDatasetRep *rep, QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_item = new RGBQGLCUDAImageItem(rep->fixedRGBLayersFromDataset()->isoSurfaceHolder(),
			rep->fixedRGBLayersFromDataset()->image(), 0,
			parent);
	m_item->setZValue(defaultZDepth);
	connect(m_rep->fixedRGBLayersFromDataset()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->fixedRGBLayersFromDataset()->image(),
			SIGNAL(rangeChanged(unsigned int , const QVector2D & )), this,
			SLOT(refresh()));

	connect(m_rep->fixedRGBLayersFromDataset()->image()->get(0), SIGNAL(dataChanged()), this,
			SLOT(refresh()));
	connect(m_rep->fixedRGBLayersFromDataset()->image()->get(1), SIGNAL(dataChanged()), this,
			SLOT(refresh()));
	connect(m_rep->fixedRGBLayersFromDataset()->image()->get(2), SIGNAL(dataChanged()), this,
			SLOT(refresh()));
}

FixedRGBLayersFromDatasetLayerOnMap::~FixedRGBLayersFromDatasetLayerOnMap() {
}

void FixedRGBLayersFromDatasetLayerOnMap::show() {
	m_scene->addItem(m_item);

}
void FixedRGBLayersFromDatasetLayerOnMap::hide() {
	m_scene->removeItem(m_item);
}

QRectF FixedRGBLayersFromDatasetLayerOnMap::boundingRect() const {
	return m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder()->worldExtent();
}
void FixedRGBLayersFromDatasetLayerOnMap::refresh() {
	m_item->update();
}


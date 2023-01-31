#include "fixedlayersfromdatasetandcubelayeronmap.h"
#include <QGraphicsScene>
#include "rgtqglcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "fixedlayersfromdatasetandcuberep.h"
#include "fixedlayersfromdatasetandcube.h"
#include "cpuimagepaletteholder.h"

FixedLayersFromDatasetAndCubeLayerOnMap::FixedLayersFromDatasetAndCubeLayerOnMap(FixedLayersFromDatasetAndCubeRep *rep, QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_item = new RGTQGLCUDAImageItem(rep->fixedLayersFromDataset()->isoSurfaceHolder(),
			rep->fixedLayersFromDataset()->image(),
			parent);
	m_item->setZValue(defaultZDepth);
	connect(m_rep->fixedLayersFromDataset()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->fixedLayersFromDataset()->image(),
			SIGNAL(rangeChanged(const QVector2D & )), this,
			SLOT(refresh()));
	connect(m_rep->fixedLayersFromDataset()->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), this,
			SLOT(refresh()));

	connect(m_rep->fixedLayersFromDataset()->image(), SIGNAL(dataChanged()), this,
			SLOT(refresh()));
}

FixedLayersFromDatasetAndCubeLayerOnMap::~FixedLayersFromDatasetAndCubeLayerOnMap() {
	hide();
}

void FixedLayersFromDatasetAndCubeLayerOnMap::show() {
	m_scene->addItem(m_item);

}
void FixedLayersFromDatasetAndCubeLayerOnMap::hide() {
	if ( m_scene )
		m_scene->removeItem(m_item);
}

QRectF FixedLayersFromDatasetAndCubeLayerOnMap::boundingRect() const {
	return m_rep->fixedLayersFromDataset()->isoSurfaceHolder()->worldExtent();
}
void FixedLayersFromDatasetAndCubeLayerOnMap::refresh() {
	m_item->update();
}


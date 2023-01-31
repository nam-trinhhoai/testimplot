#include "fixedrgblayersfromdatasetandcubelayeronmap.h"
#include <QGraphicsScene>
#include "rgbinterleavedqglcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "cudargbinterleavedimage.h"
#include "fixedrgblayersfromdatasetandcuberep.h"
#include "fixedrgblayersfromdatasetandcube.h"

FixedRGBLayersFromDatasetAndCubeLayerOnMap::FixedRGBLayersFromDatasetAndCubeLayerOnMap(FixedRGBLayersFromDatasetAndCubeRep *rep, QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_item = new RGBInterleavedQGLCUDAImageItem(rep->fixedRGBLayersFromDataset()->isoSurfaceHolder(),
			rep->fixedRGBLayersFromDataset()->image(), 0,
			parent);
	m_item->setZValue(defaultZDepth);
	connect(m_rep->fixedRGBLayersFromDataset()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->fixedRGBLayersFromDataset()->image(),
			SIGNAL(rangeChanged(unsigned int , const QVector2D & )), this,
			SLOT(refresh()));

	connect(m_rep->fixedRGBLayersFromDataset()->image(), SIGNAL(dataChanged()), this,
			SLOT(refresh()));

	connect(m_rep->fixedRGBLayersFromDataset(), SIGNAL(minimumValueActivated(bool)), this, SLOT(minValueActivated(bool)));
	connect(m_rep->fixedRGBLayersFromDataset(), SIGNAL(minimumValueChanged(float)), this, SLOT(minValueChanged(float)));
}

FixedRGBLayersFromDatasetAndCubeLayerOnMap::~FixedRGBLayersFromDatasetAndCubeLayerOnMap() {
	hide();
}

void FixedRGBLayersFromDatasetAndCubeLayerOnMap::show() {
	m_scene->addItem(m_item);

}
void FixedRGBLayersFromDatasetAndCubeLayerOnMap::hide() {
	if ( m_scene )
		m_scene->removeItem(m_item);
}

QRectF FixedRGBLayersFromDatasetAndCubeLayerOnMap::boundingRect() const {
	return m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder()->worldExtent();
}
void FixedRGBLayersFromDatasetAndCubeLayerOnMap::refresh() {
	m_item->update();
}

void FixedRGBLayersFromDatasetAndCubeLayerOnMap::minValueActivated(bool activated) {
	m_item->setMinimumValueActive(activated);
	refresh();
}

void FixedRGBLayersFromDatasetAndCubeLayerOnMap::minValueChanged(float value) {
	m_item->setMinimumValue(value);
	refresh();
}


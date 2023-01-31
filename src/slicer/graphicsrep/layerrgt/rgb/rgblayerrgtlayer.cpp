#include "rgblayerrgtlayer.h"
#include <QGraphicsScene>
#include "rgbqglcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "rgblayerrgtrep.h"
#include "rgblayerslice.h"
#include "LayerSlice.h"

RGBLayerRGTLayer::RGBLayerRGTLayer(RGBLayerRGTRep *rep, QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_item = new RGBQGLCUDAImageItem(rep->rgbLayerSlice()->layerSlice()->isoSurfaceHolder(),
			rep->rgbLayerSlice()->image(), 0,
			parent);
	m_item->setZValue(defaultZDepth);
	connect(m_rep->rgbLayerSlice()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->rgbLayerSlice()->image(),
			SIGNAL(rangeChanged(unsigned int , const QVector2D & )), this,
			SLOT(refresh()));

	connect(m_rep->rgbLayerSlice(), SIGNAL(minimumValueActivated(bool)), this, SLOT(minValueActivated(bool)));
	connect(m_rep->rgbLayerSlice(), SIGNAL(minimumValueChanged(float)), this, SLOT(minValueChanged(float)));
}

RGBLayerRGTLayer::~RGBLayerRGTLayer() {
	if (m_item->scene()!=nullptr) {
		hide();
	}
	m_item->deleteLater();
}

void RGBLayerRGTLayer::show() {
	m_scene->addItem(m_item);

}
void RGBLayerRGTLayer::hide() {
	m_scene->removeItem(m_item);
}

QRectF RGBLayerRGTLayer::boundingRect() const {
	return m_rep->rgbLayerSlice()->layerSlice()->isoSurfaceHolder()->worldExtent();
}
void RGBLayerRGTLayer::refresh() {
	m_item->update();
}

void RGBLayerRGTLayer::minValueActivated(bool activated) {
	m_item->setMinimumValueActive(activated);
	refresh();
}

void RGBLayerRGTLayer::minValueChanged(float value) {
	m_item->setMinimumValue(value);
	refresh();
}


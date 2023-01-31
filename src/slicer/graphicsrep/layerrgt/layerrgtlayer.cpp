#include "layerrgtlayer.h"

#include <QGraphicsScene>
#include "rgtqglcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "layerrgtrep.h"
#include "qglimagefilledhistogramitem.h"
#include "LayerSlice.h"

LayerRGTLayer::LayerRGTLayer(LayerRGTRep *rep, QGraphicsScene *scene,
		int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_histoItem = nullptr;
	m_item = new RGTQGLCUDAImageItem(rep->layerSlice()->isoSurfaceHolder(),
			rep->layerSlice()->image(), parent);
	m_item->setZValue(defaultZDepth);
	connect(m_rep->layerSlice()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->layerSlice()->image(), SIGNAL(rangeChanged(const QVector2D &)), this,
			SLOT(refresh()));
	connect(m_rep->layerSlice()->image(), SIGNAL(lookupTableChanged(const LookupTable &)), this,
				SLOT(refresh()));
}

LayerRGTLayer::~LayerRGTLayer() {
}

void LayerRGTLayer::showCrossHair(bool val) {
	if (val) {
		m_histoItem = new QGLImageFilledHistogramItem(m_rep->layerSlice()->image(),
				m_rep->layerSlice()->image());
		m_histoItem->setZValue(m_defaultZDepth);
		m_scene->addItem(m_histoItem);
	} else {
		if (m_histoItem != nullptr)
			m_scene->removeItem(m_histoItem);
		m_histoItem = nullptr;
	}
}

void LayerRGTLayer::mouseMoved(double worldX, double worldY, Qt::MouseButton button,
		Qt::KeyboardModifiers keys) {
	if (!m_histoItem)
		return;
	m_histoItem->mouseMove(worldX, worldY);
}


void LayerRGTLayer::show() {
	m_scene->addItem(m_item);
	if (m_histoItem)
		m_scene->addItem(m_histoItem);

}
void LayerRGTLayer::hide() {
	m_scene->removeItem(m_item);
	if (m_histoItem)
		m_scene->removeItem(m_histoItem);
}

QRectF LayerRGTLayer::boundingRect() const {
	return m_rep->layerSlice()->image()->worldExtent();
}

void LayerRGTLayer::refresh() {
	m_item->update();
}


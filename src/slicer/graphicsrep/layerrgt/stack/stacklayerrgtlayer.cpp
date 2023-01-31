#include "stacklayerrgtlayer.h"

#include <QGraphicsScene>
#include "rgtqglcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "stacklayerrgtrep.h"
#include "qglimagefilledhistogramitem.h"
#include "LayerSlice.h"

StackLayerRGTLayer::StackLayerRGTLayer(StackLayerRGTRep *rep, QGraphicsScene *scene,
		int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_histoItem = nullptr;
	m_item = new RGTQGLCUDAImageItem(rep->isoSurfaceHolder(),
			rep->image(), parent);
	m_item->setZValue(defaultZDepth);
	connect(m_rep->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->image(), SIGNAL(rangeChanged(const QVector2D &)), this,
			SLOT(refresh()));
	connect(m_rep->image(), SIGNAL(lookupTableChanged(const LookupTable &)), this,
				SLOT(refresh()));
}

StackLayerRGTLayer::~StackLayerRGTLayer() {
}

void StackLayerRGTLayer::showCrossHair(bool val) {
	if (val) {
		m_histoItem = new QGLImageFilledHistogramItem(m_rep->image(),
				m_rep->image());
		m_histoItem->setZValue(m_defaultZDepth);
		m_scene->addItem(m_histoItem);
	} else {
		if (m_histoItem != nullptr)
			m_scene->removeItem(m_histoItem);
		m_histoItem = nullptr;
	}
}

void StackLayerRGTLayer::mouseMoved(double worldX, double worldY, Qt::MouseButton button,
		Qt::KeyboardModifiers keys) {
	if (!m_histoItem)
		return;
	m_histoItem->mouseMove(worldX, worldY);
}


void StackLayerRGTLayer::show() {
	m_scene->addItem(m_item);
	if (m_histoItem)
		m_scene->addItem(m_histoItem);

}
void StackLayerRGTLayer::hide() {
	m_scene->removeItem(m_item);
	if (m_histoItem)
		m_scene->removeItem(m_histoItem);
}

QRectF StackLayerRGTLayer::boundingRect() const {
	return m_rep->layerSlice()->image()->worldExtent();
}

void StackLayerRGTLayer::refresh() {
	m_item->update();
}


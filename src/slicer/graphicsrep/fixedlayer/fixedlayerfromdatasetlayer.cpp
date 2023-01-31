#include "fixedlayerfromdatasetlayer.h"

#include <QGraphicsScene>
#include "rgtqglcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "fixedlayerfromdatasetrep.h"
#include "qglimagefilledhistogramitem.h"
#include "fixedlayerfromdataset.h"

FixedLayerFromDatasetLayer::FixedLayerFromDatasetLayer(FixedLayerFromDatasetRep *rep, QGraphicsScene *scene,
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

FixedLayerFromDatasetLayer::~FixedLayerFromDatasetLayer() {
}

void FixedLayerFromDatasetLayer::showCrossHair(bool val) {
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

void FixedLayerFromDatasetLayer::mouseMoved(double worldX, double worldY, Qt::MouseButton button,
		Qt::KeyboardModifiers keys) {
	if (!m_histoItem)
		return;
	m_histoItem->mouseMove(worldX, worldY);
}


void FixedLayerFromDatasetLayer::show() {
	m_scene->addItem(m_item);
	if (m_histoItem)
		m_scene->addItem(m_histoItem);

}
void FixedLayerFromDatasetLayer::hide() {
	m_scene->removeItem(m_item);
	if (m_histoItem)
		m_scene->removeItem(m_histoItem);
}

QRectF FixedLayerFromDatasetLayer::boundingRect() const {
	return m_rep->image()->worldExtent();
}

void FixedLayerFromDatasetLayer::refresh() {
	m_item->update();
}


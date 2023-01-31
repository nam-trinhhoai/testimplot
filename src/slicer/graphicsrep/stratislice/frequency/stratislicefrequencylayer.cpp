#include "stratislicefrequencylayer.h"
#include <QGraphicsScene>
#include "rgtqglcudaimageitem.h"
#include "stratislicefrequencyrep.h"
#include "frequencystratisliceattribute.h"
#include "cudaimagepaletteholder.h"
#include "qglimagefilledhistogramitem.h"
#include "stratislice.h"

StratiSliceFrequencyLayer::StratiSliceFrequencyLayer(StratiSliceFrequencyRep *rep, QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_histoItem = nullptr;
	m_item = new RGTQGLCUDAImageItem(rep->stratiSliceAttribute()->isoSurfaceHolder(),
			rep->stratiSliceAttribute()->image(), parent);
	m_item->setZValue(defaultZDepth);
	connect(m_rep->stratiSliceAttribute()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->stratiSliceAttribute()->image(), SIGNAL(rangeChanged(const QVector2D &)), this,
			SLOT(refresh()));
	connect(m_rep->stratiSliceAttribute()->image(), SIGNAL(lookupTableChanged(const LookupTable &)), this,
				SLOT(refresh()));
}

StratiSliceFrequencyLayer::~StratiSliceFrequencyLayer() {
}

void StratiSliceFrequencyLayer::showCrossHair(bool val) {
	if (val) {
		m_histoItem = new QGLImageFilledHistogramItem(m_rep->stratiSliceAttribute()->image(),
				m_rep->stratiSliceAttribute()->image());
		m_histoItem->setZValue(m_defaultZDepth);
		m_scene->addItem(m_histoItem);
	} else {
		if (m_histoItem != nullptr)
			m_scene->removeItem(m_histoItem);
		m_histoItem = nullptr;
	}
}

void StratiSliceFrequencyLayer::mouseMoved(double worldX, double worldY, Qt::MouseButton button,
		Qt::KeyboardModifiers keys) {
	if (!m_histoItem)
		return;
	m_histoItem->mouseMove(worldX, worldY);
}


void StratiSliceFrequencyLayer::show() {
	m_scene->addItem(m_item);
	if (m_histoItem)
		m_scene->addItem(m_histoItem);

}
void StratiSliceFrequencyLayer::hide() {
	m_scene->removeItem(m_item);
	if (m_histoItem)
		m_scene->removeItem(m_histoItem);
}

QRectF StratiSliceFrequencyLayer::boundingRect() const {
	return m_rep->stratiSliceAttribute()->image()->worldExtent();
}

void StratiSliceFrequencyLayer::refresh() {
	m_item->update();
}


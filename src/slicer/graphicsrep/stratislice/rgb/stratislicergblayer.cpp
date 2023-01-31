#include "stratislicergblayer.h"
#include <QGraphicsScene>
#include "rgbqglcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "stratislicergbattributerep.h"
#include "rgbstratisliceattribute.h"

StratiSliceRGBLayer::StratiSliceRGBLayer(StratiSliceRGBAttributeRep *rep, QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_item = new RGBQGLCUDAImageItem(rep->stratiSliceAttribute()->isoSurfaceHolder(),
			rep->stratiSliceAttribute()->image(), rep->stratiSliceAttribute()->extractionWindow(),
			parent);
	m_item->setZValue(defaultZDepth);
	connect(m_rep->stratiSliceAttribute()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->stratiSliceAttribute()->image(),
			SIGNAL(rangeChanged(unsigned int , const QVector2D & )), this,
			SLOT(refresh()));
}

StratiSliceRGBLayer::~StratiSliceRGBLayer() {
}

void StratiSliceRGBLayer::show() {
	m_scene->addItem(m_item);

}
void StratiSliceRGBLayer::hide() {
	m_scene->removeItem(m_item);
}

QRectF StratiSliceRGBLayer::boundingRect() const {
	return m_rep->stratiSliceAttribute()->isoSurfaceHolder()->worldExtent();
}
void StratiSliceRGBLayer::refresh() {
	m_item->update();
}


#include "stratisliceattributeslicelayer.h"
#include <QGraphicsScene>
#include "stratisliceattributereponslice.h"
#include "abstractstratisliceattribute.h"
#include "qglisolineitem.h"
#include "cudaimagepaletteholder.h"

StratiSliceAttributeSliceLayer::StratiSliceAttributeSliceLayer(StratiSliceAttributeRepOnSlice *rep, SliceDirection dir,
		const IGeorefImage *const transfoProvider, int startValue,
		QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth), m_transfoProvider(transfoProvider) {
	m_rep = rep;

	m_lineItem = new QGLIsolineItem(transfoProvider,
			m_rep->stratiSliceAttribute()->isoSurfaceHolder(),
			m_rep->stratiSliceAttribute()->extractionWindow(), dir, parent);
	m_lineItem->setZValue(defaultZDepth);

	m_lineItem->updateWindowSize(m_rep->stratiSliceAttribute()->extractionWindow());
	m_lineItem->updateSlice(startValue);
	m_lineItem->setColor(Qt::red);

	connect(m_rep->stratiSliceAttribute(), SIGNAL(extractionWindowChanged(uint)),
			m_lineItem, SLOT(updateWindowSize(uint)));

	connect(m_rep->stratiSliceAttribute(), SIGNAL(RGTIsoValueChanged(int)), m_lineItem,
			SLOT(updateRGTPosition()));
}

StratiSliceAttributeSliceLayer::~StratiSliceAttributeSliceLayer() {
}

void StratiSliceAttributeSliceLayer::setSliceIJPosition(int imageVal) {
	m_lineItem->updateSlice(imageVal);
}

void StratiSliceAttributeSliceLayer::show() {
	m_scene->addItem(m_lineItem);
}
void StratiSliceAttributeSliceLayer::hide() {
	m_scene->removeItem(m_lineItem);
}

QRectF StratiSliceAttributeSliceLayer::boundingRect() const {
	return m_lineItem->boundingRect();
}

void StratiSliceAttributeSliceLayer::refresh() {
	m_lineItem->update();
}

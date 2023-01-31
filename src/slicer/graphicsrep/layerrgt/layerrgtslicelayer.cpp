#include "layerrgtslicelayer.h"

#include <QGraphicsScene>
#include "layerrgtreponslice.h"
#include "cudaimagepaletteholder.h"
#include "LayerSlice.h"
#include "qglisolineitem.h"

LayerRGTSliceLayer::LayerRGTSliceLayer(LayerRGTRepOnSlice *rep, SliceDirection dir,
		const IGeorefImage *const transfoProvider, int startValue,
		QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth), m_transfoProvider(transfoProvider) {
	m_rep = rep;

	m_lineItem = new QGLIsolineItem(transfoProvider,
			m_rep->layerSlice()->isoSurfaceHolder(),
			m_rep->layerSlice()->extractionWindow(), dir, parent);
	m_lineItem->setZValue(defaultZDepth);

	m_lineItem->updateWindowSize(m_rep->layerSlice()->extractionWindow());
	m_lineItem->updateSlice(startValue);
	m_lineItem->setColor(Qt::red);

	connect(m_rep->layerSlice(), SIGNAL(extractionWindowChanged(uint)),
			m_lineItem, SLOT(updateWindowSize(uint)));

	connect(m_rep->layerSlice(), SIGNAL(RGTIsoValueChanged(int)), m_lineItem,
			SLOT(updateRGTPosition()));
}

LayerRGTSliceLayer::~LayerRGTSliceLayer() {
}

void LayerRGTSliceLayer::setSliceIJPosition(int imageVal) {
	m_lineItem->updateSlice(imageVal);
}

void LayerRGTSliceLayer::show() {
	m_scene->addItem(m_lineItem);
}
void LayerRGTSliceLayer::hide() {
	m_scene->removeItem(m_lineItem);
}

QRectF LayerRGTSliceLayer::boundingRect() const {
	return m_lineItem->boundingRect();
}

void LayerRGTSliceLayer::refresh() {
	m_lineItem->update();
}

#include "wellheadreponmap.h"
#include "wellhead.h"
#include "wellheadlayeronmap.h"
#include "abstractinnerview.h"
#include "abstract2Dinnerview.h"

WellHeadRepOnMap::WellHeadRepOnMap(WellHead *wellHead, AbstractInnerView *parent) :
	AbstractGraphicRep(parent) {
	m_data = wellHead;
	m_layer = nullptr;
}

WellHeadRepOnMap::~WellHeadRepOnMap() {
	if (m_layer != nullptr) {
		disconnect(dynamic_cast<Abstract2DInnerView*>(m_parent), &Abstract2DInnerView::viewAreaChanged,
						m_layer, &WellHeadLayerOnMap::updateFromZoom);
		delete m_layer;
	}
}

WellHead* WellHeadRepOnMap::wellHead() const {
	return m_data;
}

IData* WellHeadRepOnMap::data() const {
	return m_data;
}

QString WellHeadRepOnMap::name() const {
	return m_data->name();
}

QWidget* WellHeadRepOnMap::propertyPanel() {
	return nullptr;
}

WellHeadLayerOnMap* WellHeadRepOnMap::layer(){
	return m_layer;
}

GraphicLayer * WellHeadRepOnMap::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)
{
	if (m_layer == nullptr) {
		m_layer = new WellHeadLayerOnMap(this,scene,defaultZDepth,parent);
		connect(dynamic_cast<Abstract2DInnerView*>(m_parent), &Abstract2DInnerView::viewAreaChanged,
				m_layer, &WellHeadLayerOnMap::updateFromZoom);
	}

	return m_layer;
}

AbstractGraphicRep::TypeRep WellHeadRepOnMap::getTypeGraphicRep() {
    return Courbe;
}

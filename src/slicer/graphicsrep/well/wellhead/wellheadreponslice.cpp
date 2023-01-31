#include "wellheadreponslice.h"
#include "wellhead.h"
#include "wellheadlayeronslice.h"
#include "abstractinnerview.h"
#include "abstractsectionview.h"

WellHeadRepOnSlice::WellHeadRepOnSlice(WellHead *wellHead, AbstractInnerView *parent) :
	AbstractGraphicRep(parent) {
	m_data = wellHead;
	m_layer = nullptr;

	AbstractSectionView* view = dynamic_cast<AbstractSectionView*>(parent);
	if (view) {
		m_displayDistance = view->displayDistance();

		connect(view, &AbstractSectionView::displayDistanceChanged, this, &WellHeadRepOnSlice::setDisplayDistance);
	} else {
		m_displayDistance = 100;
	}
}

WellHeadRepOnSlice::~WellHeadRepOnSlice() {
	if (m_layer != nullptr)
		delete m_layer;
}


IData* WellHeadRepOnSlice::data() const {
	return m_data;
}

QString WellHeadRepOnSlice::name() const {
	return m_data->name();
}

QWidget* WellHeadRepOnSlice::propertyPanel() {
	return nullptr;
}
GraphicLayer * WellHeadRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)
{
	if (m_layer == nullptr) {
		m_layer = new WellHeadLayerOnSlice(this,scene,defaultZDepth,parent);
//		connect(m_data, &WellHead::displayDistanceChanged, m_layer, &WellHeadLayerOnSlice::reloadItems);
	}

	return m_layer;
}

void WellHeadRepOnSlice::setSliceIJPosition(int val) {
	if (m_layer!=nullptr) {
		m_layer->reloadItems();
	}
}

double WellHeadRepOnSlice::displayDistance() const {
	return m_displayDistance;
}

void WellHeadRepOnSlice::setDisplayDistance(double val) {
	if (m_displayDistance!=val) {
		m_displayDistance = val;
		if (m_layer) {
			m_layer->reloadItems();
		}
	}
}

AbstractGraphicRep::TypeRep WellHeadRepOnSlice::getTypeGraphicRep() {
	return Courbe;
}

void WellHeadRepOnSlice::deleteLayer(){
    if (m_layer!=nullptr) {
        delete m_layer;
        m_layer = nullptr;
    }
}

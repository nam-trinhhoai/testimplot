#include "wellheadrepnodisplay.h"
#include "wellhead.h"

WellHeadRepNoDisplay::WellHeadRepNoDisplay(WellHead *wellHead, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = wellHead;
	m_name = wellHead->name();
}

WellHeadRepNoDisplay::~WellHeadRepNoDisplay() {

}

QWidget* WellHeadRepNoDisplay::propertyPanel() {
	return nullptr;
}

GraphicLayer * WellHeadRepNoDisplay::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

IData* WellHeadRepNoDisplay::data() const {
	return m_data;
}

AbstractGraphicRep::TypeRep WellHeadRepNoDisplay::getTypeGraphicRep() {
	return AbstractGraphicRep::NotDefined;
}

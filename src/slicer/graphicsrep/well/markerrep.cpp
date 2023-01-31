#include "markerrep.h"
#include "marker.h"

MarkerRep::MarkerRep(Marker *marker, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = marker;
	m_name = marker->name();
}

MarkerRep::~MarkerRep() {

}

QWidget* MarkerRep::propertyPanel() {
	return nullptr;
}

GraphicLayer * MarkerRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

IData* MarkerRep::data() const {
	return m_data;
}

AbstractGraphicRep::TypeRep MarkerRep::getTypeGraphicRep() {
    return AbstractGraphicRep::NotDefined;
}

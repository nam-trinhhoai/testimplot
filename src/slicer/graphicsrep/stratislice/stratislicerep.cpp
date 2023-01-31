#include "stratislicerep.h"
#include "stratislice.h"

StratiSliceRep::StratiSliceRep(StratiSlice *survey,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_stratislice = survey;
	m_name = m_stratislice->name();
}

StratiSliceRep::~StratiSliceRep() {

}
IData* StratiSliceRep::data() const {
	return m_stratislice;
}

QWidget* StratiSliceRep::propertyPanel() {
	return nullptr;
}
GraphicLayer* StratiSliceRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	return nullptr;
}

AbstractGraphicRep::TypeRep StratiSliceRep::getTypeGraphicRep() {
    return AbstractGraphicRep::NotDefined;
}

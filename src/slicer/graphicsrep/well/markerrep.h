#ifndef MarkerRep_H
#define MarkerRep_H

#include <QObject>
#include "abstractgraphicrep.h"
class Marker;

class MarkerRep: public AbstractGraphicRep {
Q_OBJECT
public:
	MarkerRep(Marker *marker, AbstractInnerView *parent = 0);
	virtual ~MarkerRep();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return false;
	}

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	virtual IData* data() const override;
	virtual AbstractGraphicRep::TypeRep getTypeGraphicRep() override;

private:
	Marker *m_data;
};

#endif

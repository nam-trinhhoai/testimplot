#ifndef StratiSliceRep_H
#define StratiSliceRep_H

#include <QObject>
#include "abstractgraphicrep.h"
class StratiSlice;

class StratiSliceRep: public AbstractGraphicRep {
Q_OBJECT
public:
	StratiSliceRep(StratiSlice *stratislice, AbstractInnerView *parent = 0);
	virtual ~StratiSliceRep();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return false;
	}

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	virtual IData* data() const override;
	virtual TypeRep getTypeGraphicRep() override;

protected:
	StratiSlice *m_stratislice;
};

#endif

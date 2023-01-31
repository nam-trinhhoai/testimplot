#ifndef WellHeadRepNoDisplay_H
#define WellHeadRepNoDisplay_H

#include <QObject>
#include "abstractgraphicrep.h"
class WellHead;

class WellHeadRepNoDisplay: public AbstractGraphicRep {
Q_OBJECT
public:
	WellHeadRepNoDisplay(WellHead *wellHead, AbstractInnerView *parent = 0);
	virtual ~WellHeadRepNoDisplay();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return false;
	}

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	virtual IData* data() const override;
	virtual TypeRep getTypeGraphicRep() override;
private:
	WellHead *m_data;
};

#endif

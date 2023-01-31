#ifndef WellHeadRepOnSlice_H
#define WellHeadRepOnSlice_H

#include <QObject>
#include <QMap>

#include "isliceablerep.h"
#include "abstractgraphicrep.h"

class WellHead;
class WellHeadLayerOnSlice;

class WellHeadRepOnSlice: public AbstractGraphicRep, public ISliceableRep {
Q_OBJECT
public:
	WellHeadRepOnSlice(WellHead *wellHead, AbstractInnerView *parent = 0);
	virtual ~WellHeadRepOnSlice();

	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override {
		return false;
	}
	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	virtual void setSliceIJPosition(int val) override;
	virtual TypeRep getTypeGraphicRep() override;
	virtual void deleteLayer() override;
	double displayDistance() const;
	void setDisplayDistance(double);


private:
	WellHeadLayerOnSlice *m_layer;
	WellHead* m_data;

	double m_displayDistance;
};

#endif

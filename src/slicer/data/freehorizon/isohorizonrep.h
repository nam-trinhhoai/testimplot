

#ifndef __ISOHORIZONREP__
#define __ISOHORIZONREP__

#include <QObject>
#include <QMenu>

#include "abstractgraphicrep.h"
class IsoHorizon;

class IsoHorizonRep: public AbstractGraphicRep {
Q_OBJECT
public:
	IsoHorizonRep(IsoHorizon *freehorizon, AbstractInnerView *parent = 0);
	virtual ~IsoHorizonRep();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return false;
	}

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	virtual IData* data() const override;
	virtual AbstractGraphicRep::TypeRep getTypeGraphicRep() override;
	virtual void buildContextMenu(QMenu *menu) override;

private:
	IsoHorizon *m_data;

signals:
	void deletedRep(AbstractGraphicRep *rep);

private slots:
	void deleteHorizon();

};

#endif













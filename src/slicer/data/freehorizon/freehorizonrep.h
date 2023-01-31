

#ifndef __FREEHORIZONREP__
#define __FREEHORIZONREP__

#include <QObject>
#include <QMenu>
#include "abstractgraphicrep.h"
class FreeHorizon;

class FreeHorizonRep: public AbstractGraphicRep {
Q_OBJECT
public:
	FreeHorizonRep(FreeHorizon *freehorizon, AbstractInnerView *parent = 0);
	virtual ~FreeHorizonRep();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return false;
	}

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	virtual IData* data() const override;
	virtual AbstractGraphicRep::TypeRep getTypeGraphicRep() override;
	void buildContextMenu(QMenu *menu) override;

signals:
	void deletedRep(AbstractGraphicRep *rep);


private:
	FreeHorizon *m_data = nullptr;

private slots:
	void computeAttribut();
	void deleteHorizon();
	void infoHorizon();
	void folderHorizon();
	void exportToSismage();


};

#endif













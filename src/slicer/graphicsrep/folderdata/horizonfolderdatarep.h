#ifndef HorizonFolderDataRep_H
#define HorizonFolderDataRep_H

#include <QObject>
#include "abstractgraphicrep.h"
class HorizonFolderData;
class HorizonPropPanel;

class HorizonFolderDataRep: public AbstractGraphicRep {
Q_OBJECT
public:
HorizonFolderDataRep(HorizonFolderData *folderData, AbstractInnerView *parent = 0);
	virtual ~HorizonFolderDataRep();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return false;
	}

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	//virtual void buildContextMenu(QMenu *menu) override; // MZR 20082021
	virtual TypeRep getTypeGraphicRep() override;

	virtual IData* data() const override;


private slots:
//	void addData();
	void showHorizonWidget();
	//void addSismageHorizon();
	//void computeAttributHorizon();

private:
	HorizonFolderData *m_data;
	HorizonPropPanel *m_propPanel = nullptr;



};

#endif

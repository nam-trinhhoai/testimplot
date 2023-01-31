#ifndef FreeHorizonTreeWidgetItem_H
#define FreeHorizonTreeWidgetItem_H

#include <QTreeWidgetItem>
#include "datatreewidgetitem.h"

class FreeHorizon;
class FixedRGBLayersFromDatasetAndCube;

//Shortcut: this double inheriting is not really good as QObject mecanism are expensive.
//In case of too many nodes, A custom model should be implemented
class FreeHorizonTreeWidgetItem:  public DataTreeWidgetItem {
	Q_OBJECT
public:
	FreeHorizonTreeWidgetItem(FreeHorizon* freeHorizon,const std::vector<ViewType>& viewType,QTreeWidgetItem *parent=0, QObject* parentObj=0);
	virtual ~FreeHorizonTreeWidgetItem();

	const FreeHorizon* getFreeHorizon() const;
	FreeHorizon* getFreeHorizon();

private slots:
	//Dynamic rep insertion
	void insertChildBore(FixedRGBLayersFromDatasetAndCube* wellBore);
	//Data changed
	//void childAdded(IGraphicRepFactory *child);

private:
	FreeHorizon *m_freeHorizon;
};

#endif

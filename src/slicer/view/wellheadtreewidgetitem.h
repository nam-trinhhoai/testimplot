#ifndef WellHeadTreeWidgetItem_H
#define WellHeadTreeWidgetItem_H

#include <QTreeWidgetItem>
#include "datatreewidgetitem.h"

class WellHead;
class WellBore;

//Shortcut: this double inheriting is not really good as QObject mecanism are expensive.
//In case of too many nodes, A custom model should be implemented
class WellHeadTreeWidgetItem:  public DataTreeWidgetItem {
	Q_OBJECT
public:
	WellHeadTreeWidgetItem(WellHead* wellHead,const std::vector<ViewType>& viewType,QTreeWidgetItem *parent=0, QObject* parentObj=0);
	virtual ~WellHeadTreeWidgetItem();

	const WellHead* getWellHead() const;
	WellHead* getWellHead();

private slots:
	//Dynamic rep insertion
	void insertChildBore(WellBore* wellBore);
	//Data changed
	//void childAdded(IGraphicRepFactory *child);

private:
	WellHead *m_wellHead;
};

#endif

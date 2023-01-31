#ifndef MarkerTreeWidgetItem_H
#define MarkerTreeWidgetItem_H

#include <QTreeWidgetItem>
#include "datatreewidgetitem.h"

class Marker;
class WellPick;

//Shortcut: this double inheriting is not really good as QObject mecanism are expensive.
//In case of too many nodes, A custom model should be implemented
class MarkerTreeWidgetItem:  public DataTreeWidgetItem {
	Q_OBJECT
public:
	MarkerTreeWidgetItem(Marker* marker,const std::vector<ViewType>& viewTypes,QTreeWidgetItem *parent=0, QObject* parentObj=0);
	virtual ~MarkerTreeWidgetItem();

	const Marker* getMarker() const;
	Marker* getMarker();

private slots:
	//Dynamic rep insertion
	void insertChildPick(WellPick* pick);
	//Data changed
	//void childAdded(IGraphicRepFactory *child);

private:
	Marker *m_marker;
};

#endif

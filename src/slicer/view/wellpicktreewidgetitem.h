#ifndef WellPickTreeWidgetItem_H_
#define WellPickTreeWidgetItem_H_

#include "datatreewidgetitem.h"

class WellPick;

class WellPickTreeWidgetItem: public DataTreeWidgetItem {
	Q_OBJECT
public:
	WellPickTreeWidgetItem(WellPick* wellPick,const std::vector<ViewType>& viewTypes,QTreeWidgetItem *parent, QObject* parentObj=0);
	virtual ~WellPickTreeWidgetItem();

	const WellPick* getWellPick() const;
	WellPick* getWellPick();

private:
	WellPick* m_wellPick;
};

#endif

#ifndef DataTreeWidgetItem_H
#define DataTreeWidgetItem_H

#include "viewutils.h"

#include <QTreeWidgetItem>

class IData;

//Shortcut: this double inheriting is not really good as QObject mecanism are expensive.
//In case of too many nodes, A custom model should be implemented
class DataTreeWidgetItem:  public QObject ,public QTreeWidgetItem{
	Q_OBJECT
public:
	DataTreeWidgetItem(IData* data,const std::vector<ViewType>& viewTypes,QTreeWidgetItem *parent=0, QObject* parentObj=0);
	virtual ~DataTreeWidgetItem();

	const IData* getData() const;
	IData* getData();
	const std::vector<ViewType>& getViewTypes() const;

private slots:
	void dataDisplayPreferenceChanged(std::vector<ViewType>, bool);

private:
	IData *m_data;
	std::vector<ViewType> m_viewTypes;
};

#endif

#ifndef FolderDataTreeWidgetItem_H
#define FolderDataTreeWidgetItem_H

#include <QTreeWidgetItem>
#include "datatreewidgetitem.h"

class FolderData;

//Shortcut: this double inheriting is not really good as QObject mecanism are expensive.
//In case of too many nodes, A custom model should be implemented
class FolderDataTreeWidgetItem:  public DataTreeWidgetItem {
	Q_OBJECT
public:
	FolderDataTreeWidgetItem(FolderData* folder, const std::vector<ViewType>& viewTypes, QTreeWidgetItem *parent, QObject* parentObj=0);
	virtual ~FolderDataTreeWidgetItem();

	const FolderData* getFolderData() const;
	FolderData* getFolderData();

private slots:
	//Dynamic rep insertion
	virtual void insertChildData(IData* data);
	virtual void removeChildData(IData *data);
	//Data changed
	//void childAdded(IGraphicRepFactory *child);

private:
	virtual DataTreeWidgetItem* getTreeItem(IData* data); // create new item

	FolderData *m_folder;
	QList<ViewType> m_viewTypes;
};
#endif

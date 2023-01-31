#include "folderdatatreewidgetitem.h"
#include "datatreewidgetitem.h"
#include "folderdata.h"
#include "freehorizon.h"
#include "idata.h"
#include "marker.h"
#include "wellhead.h"
#include "freehorizontreewidgetitem.h"
#include "markertreewidgetitem.h"
#include "wellheadtreewidgetitem.h"
#include <QDebug>
#include <QCoreApplication>

#include <algorithm>

bool compareNames(QTreeWidgetItem* first, QTreeWidgetItem* second) {
	QString nameFirst = first->data(0, Qt::DisplayRole).toString();
	QString nameSecond = second->data(0, Qt::DisplayRole).toString();
	return nameFirst.compare(nameSecond)<0;
}

FolderDataTreeWidgetItem::FolderDataTreeWidgetItem(FolderData* folderData,
		const std::vector<ViewType>& viewTypes, QTreeWidgetItem *parent, QObject* parentObj) :
		DataTreeWidgetItem(folderData, viewTypes, parent, parentObj) {
	m_folder = folderData;
//	{
//		QSignalBlocker block(parent->treeWidget()->model());
//		setData(0, Qt::DisplayRole, QVariant::fromValue(marker->name()));
//		setData(0, Qt::UserRole, QVariant::fromValue(marker));
//		setData(0, Qt::CheckStateRole, QVariant::fromValue(marker->displayPreference()));
//	}
	connect(m_folder, SIGNAL(dataAdded(IData *)), this,
			SLOT(insertChildData(IData *)));
	connect(m_folder, SIGNAL(dataRemoved(IData *)), this,
			SLOT(removeChildData(IData *)));

	QList<IData*> datas = m_folder->data();
	QList<QTreeWidgetItem*> itemsToAdd;
	for (IData *data : datas) {
		DataTreeWidgetItem *el = getTreeItem(data);
		itemsToAdd.push_back(el);
	}

	if (itemsToAdd.count()>1) {
		std::sort(itemsToAdd.begin(), itemsToAdd.end(), compareNames);
	}

	addChildren(itemsToAdd);
	setExpanded(true);

//	connect(m_marker, &IData::displayPreferenceChanged, this, &MarkerTreeWidgetItem::dataDisplayPreferenceChanged);
}

//void FolderDataTreeWidgetItem::dataDisplayPreferenceChanged(bool newPreference) {
//	if (this->data(0, Qt::CheckStateRole).toBool()!=newPreference) {
//		Qt::CheckState state = newPreference ? Qt::Checked : Qt::Unchecked;
//		this->setCheckState(0, state);
//	}
//}

void FolderDataTreeWidgetItem::insertChildData(IData *data) {
	DataTreeWidgetItem *childNode = getTreeItem(data);
	int i = 0;
	while (i<childCount() && compareNames(child(i), childNode)) {
		i++;
	}
	insertChild(i, childNode);
}

void FolderDataTreeWidgetItem::removeChildData(IData *data) {
	DataTreeWidgetItem *childNode = nullptr;
	long i=0;
	while (childNode==nullptr && i<childCount()) {
		DataTreeWidgetItem* currentChild =
				dynamic_cast<DataTreeWidgetItem*>(this->child(i));
		if (currentChild!=nullptr && currentChild->getData()==data) {
			childNode = currentChild;
		}
		i++;
	}
	if (childNode == nullptr)
		return;
	if (childNode->checkState(0)!=Qt::Unchecked) {
		childNode->setCheckState(0, Qt::Unchecked);
		QCoreApplication::processEvents(); // to unselect
	}
	removeChild(childNode);
	childNode->deleteLater();
}

FolderDataTreeWidgetItem::~FolderDataTreeWidgetItem() {
}

const FolderData* FolderDataTreeWidgetItem::getFolderData() const {
	return m_folder;
}

FolderData* FolderDataTreeWidgetItem::getFolderData() {
	return m_folder;
}

DataTreeWidgetItem* FolderDataTreeWidgetItem::getTreeItem(IData* data) {
	DataTreeWidgetItem* out = nullptr;

	if (Marker* marker = dynamic_cast<Marker*>(data)) {
		out = new MarkerTreeWidgetItem(marker, getViewTypes());
	} else if (WellHead* wellHead = dynamic_cast<WellHead*>(data)) {
		out = new WellHeadTreeWidgetItem(wellHead, getViewTypes());
	} else if (FreeHorizon* freeHorizon = dynamic_cast<FreeHorizon*>(data)) {
		out = new FreeHorizonTreeWidgetItem(freeHorizon, getViewTypes());
	} else {
		out = new DataTreeWidgetItem(data, getViewTypes());
	}
	return out;
}

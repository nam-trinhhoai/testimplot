#include "wellheadtreewidgetitem.h"
#include "datatreewidgetitem.h"
#include "wellhead.h"
#include "wellbore.h"
#include <QDebug>

WellHeadTreeWidgetItem::WellHeadTreeWidgetItem(WellHead* wellHead,
		const std::vector<ViewType>& viewType, QTreeWidgetItem *parent, QObject* parentObj) :
		DataTreeWidgetItem(wellHead, viewType, parent, parentObj) {
	m_wellHead = wellHead;
//	{
//		QSignalBlocker block(parent->treeWidget()->model());
//		setData(0, Qt::DisplayRole, QVariant::fromValue(marker->name()));
//		setData(0, Qt::UserRole, QVariant::fromValue(marker));
//		setData(0, Qt::CheckStateRole, QVariant::fromValue(marker->displayPreference()));
//	}
	connect(m_wellHead, SIGNAL(wellBoreAdded(WellBore *)), this,
			SLOT(insertChildBore(WellBore *)));

	QList<WellBore*> bores = m_wellHead->wellBores();
	QList<QTreeWidgetItem*> itemsToAdd;
	for (WellBore *bore : bores) {
		DataTreeWidgetItem *el = new DataTreeWidgetItem(bore, getViewTypes(), this);
		itemsToAdd.push_back(el);
	}
	addChildren(itemsToAdd);
//	setExpanded(false);

//	connect(m_marker, &IData::displayPreferenceChanged, this, &MarkerTreeWidgetItem::dataDisplayPreferenceChanged);
}

//void WellHeadTreeWidgetItem::dataDisplayPreferenceChanged(bool newPreference) {
//	if (this->data(0, Qt::CheckStateRole).toBool()!=newPreference) {
//		Qt::CheckState state = newPreference ? Qt::Checked : Qt::Unchecked;
//		this->setCheckState(0, state);
//	}
//}

void WellHeadTreeWidgetItem::insertChildBore(WellBore *bore) {
	DataTreeWidgetItem *childNode = new DataTreeWidgetItem(bore, getViewTypes(), this);
	addChild(childNode);
}

WellHeadTreeWidgetItem::~WellHeadTreeWidgetItem() {
}

const WellHead* WellHeadTreeWidgetItem::getWellHead() const {
	return m_wellHead;
}

WellHead* WellHeadTreeWidgetItem::getWellHead() {
	return m_wellHead;
}

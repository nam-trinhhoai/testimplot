#include "markertreewidgetitem.h"
#include "datatreewidgetitem.h"
#include "wellpicktreewidgetitem.h"
#include "marker.h"
#include "wellpick.h"
#include <QDebug>

MarkerTreeWidgetItem::MarkerTreeWidgetItem(Marker* marker,
		const std::vector<ViewType>& viewTypes, QTreeWidgetItem *parent, QObject* parentObj) :
		DataTreeWidgetItem(marker, viewTypes, parent, parentObj) {
	m_marker = marker;
//	{
//		QSignalBlocker block(parent->treeWidget()->model());
//		setData(0, Qt::DisplayRole, QVariant::fromValue(marker->name()));
//		setData(0, Qt::UserRole, QVariant::fromValue(marker));
//		setData(0, Qt::CheckStateRole, QVariant::fromValue(marker->displayPreference()));
//	}
	connect(m_marker, SIGNAL(wellPickAdded(WellPick *)), this,
			SLOT(insertChildPick(WellPick *)));

	QList<WellPick*> picks = m_marker->wellPicks();
	QList<QTreeWidgetItem*> itemsToAdd;
	for (WellPick *pick : picks) {
		DataTreeWidgetItem *el = new WellPickTreeWidgetItem(pick, getViewTypes(), this);
		itemsToAdd.push_back(el);
	}
	addChildren(itemsToAdd);
//	setExpanded(false);

//	connect(m_marker, &IData::displayPreferenceChanged, this, &MarkerTreeWidgetItem::dataDisplayPreferenceChanged);
}

//void MarkerTreeWidgetItem::dataDisplayPreferenceChanged(bool newPreference) {
//	if (this->data(0, Qt::CheckStateRole).toBool()!=newPreference) {
//		Qt::CheckState state = newPreference ? Qt::Checked : Qt::Unchecked;
//		this->setCheckState(0, state);
//	}
//}

void MarkerTreeWidgetItem::insertChildPick(WellPick *pick) {
	DataTreeWidgetItem *childNode = new WellPickTreeWidgetItem(pick, getViewTypes(), this);
	addChild(childNode);
}

MarkerTreeWidgetItem::~MarkerTreeWidgetItem() {
}

const Marker* MarkerTreeWidgetItem::getMarker() const {
	return m_marker;
}

Marker* MarkerTreeWidgetItem::getMarker() {
	return m_marker;
}

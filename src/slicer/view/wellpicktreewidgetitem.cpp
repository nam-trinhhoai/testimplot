#include "wellpicktreewidgetitem.h"
#include "wellpick.h"

WellPickTreeWidgetItem::WellPickTreeWidgetItem(WellPick* wellPick,
		const std::vector<ViewType>& viewTypes, QTreeWidgetItem *parent, QObject* parentObj) :
		DataTreeWidgetItem(wellPick, viewTypes, parent, parentObj) {
	m_wellPick = wellPick;
	if (parent!=nullptr && parent->treeWidget()!=nullptr && parent->treeWidget()->model()!=nullptr) {
		QSignalBlocker block(parent->treeWidget()->model());
		setData(0, Qt::DisplayRole, QVariant::fromValue(wellPick->wellBore()->name()));
		setToolTip(0,wellPick->wellBore()->name());
	} else {
		setData(0, Qt::DisplayRole, QVariant::fromValue(wellPick->wellBore()->name()));
		setToolTip(0,wellPick->wellBore()->name());
	}
}


WellPickTreeWidgetItem::~WellPickTreeWidgetItem() {
}

const WellPick* WellPickTreeWidgetItem::getWellPick() const {
	return m_wellPick;
}

WellPick* WellPickTreeWidgetItem::getWellPick() {
	return m_wellPick;
}

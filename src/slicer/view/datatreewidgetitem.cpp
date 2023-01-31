#include "datatreewidgetitem.h"
#include "idata.h"

#include <QDebug>

DataTreeWidgetItem::DataTreeWidgetItem(IData* data,
		const std::vector<ViewType>& viewTypes, QTreeWidgetItem *parent, QObject* parentObj) :
		QTreeWidgetItem(parent), QObject(parentObj) {
	m_data = data;
	m_viewTypes = viewTypes;
	if (parent!=nullptr && parent->treeWidget()!=nullptr && parent->treeWidget()->model()!=nullptr) {
		QSignalBlocker block(parent->treeWidget()->model());
		setData(0, Qt::DisplayRole, QVariant::fromValue(data->name()));
		setData(0, Qt::UserRole, QVariant::fromValue(data));
		setData(0, Qt::CheckStateRole, QVariant::fromValue(data->displayPreferences(viewTypes)));
	} else {
		setData(0, Qt::DisplayRole, QVariant::fromValue(data->name()));
		setData(0, Qt::UserRole, QVariant::fromValue(data));
		setData(0, Qt::CheckStateRole, QVariant::fromValue(data->displayPreferences(viewTypes)));
	}

	connect(m_data, &IData::displayPreferenceChanged, this, &DataTreeWidgetItem::dataDisplayPreferenceChanged);
}

void DataTreeWidgetItem::dataDisplayPreferenceChanged(std::vector<ViewType> viewTypesChanged, bool preferenceChanged) {
	const std::vector<ViewType>& viewTypes = getViewTypes();
	bool update = false;
	int idx = 0;
	while (!update && idx<viewTypesChanged.size()) {
		ViewType viewTypeChanged = viewTypesChanged[idx];
		auto it = std::find(viewTypes.begin(), viewTypes.end(), viewTypeChanged);
		update = it!=viewTypes.end();
		idx++;
	}

	if (update) {
		bool newPreferences = m_data->displayPreferences(m_viewTypes);
		if (this->data(0, Qt::CheckStateRole).toBool()!=newPreferences) {
			Qt::CheckState state = newPreferences ? Qt::Checked : Qt::Unchecked;
			this->setCheckState(0, state);
		}
	}
}

DataTreeWidgetItem::~DataTreeWidgetItem() {
}

const IData* DataTreeWidgetItem::getData() const {
	return m_data;
}

IData* DataTreeWidgetItem::getData() {
	return m_data;
}

const std::vector<ViewType>& DataTreeWidgetItem::getViewTypes() const {
	return m_viewTypes;
}

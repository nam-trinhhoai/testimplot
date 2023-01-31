#include "idata.h"

#include "igraphicrepfactory.h"
#include "workingsetmanager.h"
#include <chrono>
#include <QDebug>

using namespace std::chrono;


IData::IData(WorkingSetManager *manager, QObject* parent) : QObject(parent) {
	m_manager = manager;
}

IData::~IData() {
}

WorkingSetManager* IData::workingSetManager() const {
	return m_manager;
}

bool IData::displayPreference(ViewType viewType) const {
	auto it = m_displayPreferences.find(viewType);
	bool preference = it!=m_displayPreferences.end() && it->second;
	return preference;
}

bool IData::displayPreferences(const std::vector<ViewType>& viewTypes) const {
	bool preference = true;
	int i=0;
	while (preference && i<viewTypes.size()) {
		preference = displayPreference(viewTypes[i]);
		i++;
	}
	return preference;
}

void IData::setAllDisplayPreference(bool val) {
	std::vector<ViewType> viewTypes = allViewTypes();
	setDisplayPreferences(viewTypes, val);
}

void IData::setDisplayPreference(ViewType viewType, bool val) {
	bool currentPreference = displayPreference(viewType);
	if (val!=currentPreference) {
		m_displayPreferences[viewType] = val;
	//	steady_clock::time_point begin = steady_clock::now();
		emit displayPreferenceChanged({viewType}, val);

	//	steady_clock::time_point end = steady_clock::now();
	//	duration<double> time_spanTot = duration_cast<duration<double>>(end - begin);

	// qDebug() << "creation object "<< time_spanTot.count() << " seconds.";
	}
}

void IData::setDisplayPreferences(const std::vector<ViewType>& viewTypes, bool val) {
	std::vector<ViewType> modifiedTypes;
	for (ViewType viewType : viewTypes) {
		bool currentPreference = displayPreference(viewType);
		if (val!=currentPreference) {
			m_displayPreferences[viewType] = val;
			modifiedTypes.push_back(viewType);
		}
	}
	if (modifiedTypes.size()>0) {
		emit displayPreferenceChanged(modifiedTypes, val);
	}
}

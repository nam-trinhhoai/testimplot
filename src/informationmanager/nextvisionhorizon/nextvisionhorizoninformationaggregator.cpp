#include "nextvisionhorizoninformationaggregator.h"

#include "GeotimeProjectManagerWidget.h"
#include "ProjectManagerWidget.h"
#include "nextvisionhorizoninformation.h"
#include "nextvisionhorizoninformationiterator.h"
// #include "nextvisionhorizonwidget.h"
#include "workingsetmanager.h"

#include <iterator>

void NextvisionHorizonInformationAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

NextvisionHorizonInformationAggregator::NextvisionHorizonInformationAggregator(WorkingSetManager* manager, bool enableToggleAction, QObject* parent) :
		IInformationAggregator(parent), m_manager(manager) {

	m_enableToggleAction = enableToggleAction;
	if ( manager == nullptr ) return;
	std::vector<QString> dataFullname;
	std::vector<QString> dataTinyname;

	if ( manager->getManagerWidget() )
	{
		dataFullname = manager->getManagerWidget()->get_freehorizon_fullnames();
		dataTinyname = manager->getManagerWidget()->get_freehorizon_names();
	}
	else if ( manager->getManagerWidgetV2() )
	{
		dataFullname = manager->getManagerWidgetV2()->getFreeHorizonFullName();
		dataTinyname = manager->getManagerWidgetV2()->getFreeHorizonNames();
	}

	long N = std::min(dataFullname.size(), dataTinyname.size());
	for (long i=0; i<N; i++) {
		QString name = dataTinyname[i];
		name.replace(".txt", "");
		NextvisionHorizonInformation* info = new NextvisionHorizonInformation(name, dataFullname[i], manager, m_enableToggleAction);
		insert(i, info);
	}
}

NextvisionHorizonInformationAggregator::~NextvisionHorizonInformationAggregator() {

}

QString NextvisionHorizonInformationAggregator::surveyName()
{
	WorkingSetManager *p = (WorkingSetManager*)m_manager;
	if ( !p ) return "";
	if ( p->getManagerWidget() )
	{
		return p->getManagerWidget()->get_survey_name();
	}
	if ( p->getManagerWidgetV2() )
	{
		return p->getManagerWidgetV2()->getSurveyName();
	}
	return "";
}

QString NextvisionHorizonInformationAggregator::projectName()
{
	WorkingSetManager *p = (WorkingSetManager*)m_manager;
	if ( !p ) return "";
	if ( p->getManagerWidget() )
	{
		return p->getManagerWidget()->get_projet_name();
	}
	if ( p->getManagerWidgetV2() )
	{
		return p->getManagerWidgetV2()->getProjectName();
	}
	return "";
}

bool NextvisionHorizonInformationAggregator::isCreatable() const {
	return true;
}

bool NextvisionHorizonInformationAggregator::createStorage() {
	// NextvisionHorizonWidget::showWidget();
	return true;
}

bool NextvisionHorizonInformationAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	NextvisionHorizonInformation* info = dynamic_cast<NextvisionHorizonInformation*>(information);
	if (info==nullptr) {
		return false;
	}

	bool res = info->deleteStorage(errorMsg);
	if (res) {
		long i = 0;
		long N = size();
		bool foundIdx = false;
		auto it = m_informations.begin();
		while (!foundIdx && i<N) {
			foundIdx = it->obj==info;
			if (!foundIdx) {
				i++;
				it++;
			}
		}
		if (i<N) {
			remove(i);
		}
		info->deleteLater();
	}
	return res;
}

NextvisionHorizonInformation* NextvisionHorizonInformationAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const NextvisionHorizonInformation* NextvisionHorizonInformationAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> NextvisionHorizonInformationAggregator::begin() {
	NextvisionHorizonInformationIterator* it = new NextvisionHorizonInformationIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> NextvisionHorizonInformationAggregator::end() {
	NextvisionHorizonInformationIterator* it = new NextvisionHorizonInformationIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long NextvisionHorizonInformationAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> NextvisionHorizonInformationAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE};
}

void NextvisionHorizonInformationAggregator::insert(long i, NextvisionHorizonInformation* information) {
	ListItem item;
	item.obj = information;
	item.originParent = information->parent();
	information->setParent(this);

	if (i>=size()) {
		i = size();
		m_informations.push_back(item);
	} else {
		if (i<0) {
			i = 0;
		}
		auto it = m_informations.begin();
		if (i>0) {
			std::advance(it, i);
		}
		m_informations.insert(it, item);
	}

	emit nextvisionHorizonInformationAdded(i, information);
	emit informationAdded(information);
}

void NextvisionHorizonInformationAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	NextvisionHorizonInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	emit nextvisionHorizonInformationRemoved(i, information);
	emit informationRemoved(information);
}

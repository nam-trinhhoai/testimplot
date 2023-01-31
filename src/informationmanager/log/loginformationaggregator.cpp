

#include <loginformationaggregator.h>

#include "GeotimeProjectManagerWidget.h"
#include "ProjectManagerWidget.h"
#include "loginformation.h"
#include "loginformationiterator.h"
#include <GeotimeProjectManagerWidget.h>
#include <ProjectManagerWidget.h>
// #include "nurbswidget.h"
#include "workingsetmanager.h"
#include "wellbore.h"
#include "loginformation.h"


#include <iterator>
#include <QProcess>

void LogInformationAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

LogInformationAggregator::LogInformationAggregator(WellBore* wellBore, QObject* parent) :
		IInformationAggregator(parent), m_manager(nullptr) {

	if ( wellBore == nullptr ) return;
	m_wellBore = wellBore;
	std::vector<QString> dataFullname = m_wellBore->logsFiles();
	std::vector<QString> dataTinyname = m_wellBore->logsNames();
	LogInformation* info0 = new LogInformation("None", "", m_manager);
	info0->setUserName(m_userName);
	insert(0, info0);
	long N = std::min(dataFullname.size(), dataTinyname.size());
	for (long i=0; i<N; i++) {
		QString name = dataTinyname[i];
		// name.replace(".txt", "");
		LogInformation* info = new LogInformation(name, dataFullname[i], m_manager);
		info->setUserName(m_userName);
		insert(i+1, info);
	}
	m_userName = getUserName();
}

LogInformationAggregator::~LogInformationAggregator() {

}

QString LogInformationAggregator::surveyName()
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

QString LogInformationAggregator::projectName()
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

bool LogInformationAggregator::isCreatable() const {
	return true;
}

bool LogInformationAggregator::createStorage() {
	// NurbsWidget::showWidget();
	return true;
}

bool LogInformationAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	LogInformation* info = dynamic_cast<LogInformation*>(information);
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

LogInformation* LogInformationAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const LogInformation* LogInformationAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> LogInformationAggregator::begin() {
	LogInformationIterator* it = new LogInformationIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> LogInformationAggregator::end() {
	LogInformationIterator* it = new LogInformationIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long LogInformationAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> LogInformationAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE};
}

void LogInformationAggregator::insert(long i, LogInformation* information) {
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

	// emit seismicInformationAdded(i, information);
	// emit informationAdded(information);
}

void LogInformationAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	LogInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	// emit seismicInformationRemoved(i, information);
	// emit informationRemoved(information);
}

QString LogInformationAggregator::getUserName()
{
	QProcess process;
	process.start("whoami");
	process.waitForFinished();
	QString name = "";
	if (process.exitCode()==QProcess::NormalExit) {
		name = process.readAllStandardOutput();
	} else {
		name = "unknown";
	}
	return name;
}

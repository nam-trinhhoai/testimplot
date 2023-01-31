

#include <seismicinformationaggregator.h>

#include "GeotimeProjectManagerWidget.h"
#include "ProjectManagerWidget.h"
#include "seismicinformation.h"
#include "seismicinformationiterator.h"
#include <GeotimeProjectManagerWidget.h>
#include <ProjectManagerWidget.h>
// #include "nurbswidget.h"
#include "workingsetmanager.h"

#include <iterator>
#include <QProcess>

void SeismicInformationAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

SeismicInformationAggregator::SeismicInformationAggregator(WorkingSetManager* manager, bool enableToggleAction, QObject* parent) :
		IInformationAggregator(parent), m_manager(manager) {
	m_enableToggleAction = enableToggleAction;
	if ( manager == nullptr ) return;
	std::vector<QString> dataFullname;
	std::vector<QString> dataTinyname;

	if ( manager->getManagerWidget() )
	{
		manager->getManagerWidget()->seismic_database_update();
		dataFullname = manager->getManagerWidget()->get_seismic_AllFullnames();
		dataTinyname = manager->getManagerWidget()->get_seismic_AllTinynames();
	}
	else if ( manager->getManagerWidgetV2() )
	{
		manager->getManagerWidgetV2()->seimsicDatabaseUpdate();
		dataFullname = manager->getManagerWidgetV2()->getSeismicAllPath();
		dataTinyname = manager->getManagerWidgetV2()->getSeismicAllNames();
	}

	m_userName = getUserName();

	long N = std::min(dataFullname.size(), dataTinyname.size());
	for (long i=0; i<N; i++) {
		QString name = dataTinyname[i];
		// name.replace(".txt", "");
		SeismicInformation* info = new SeismicInformation(name, dataFullname[i], manager, m_enableToggleAction);
		info->setUserName(m_userName);
		insert(i, info);
	}
}

SeismicInformationAggregator::SeismicInformationAggregator(WorkingSetManager* manager, int dimx, int dimy , int dimz, bool enableToggleAction, QObject* parent)
{
	m_enableToggleAction = enableToggleAction;
	if ( manager == nullptr ) return;
	std::vector<QString> dataFullname;
	std::vector<QString> dataTinyname;
	std::vector<int> dataDimx;
	std::vector<int> dataDimy;
	std::vector<int> dataDimz;

	if ( manager->getManagerWidget() )
	{
		manager->getManagerWidget()->seismic_database_update();
		dataFullname = manager->getManagerWidget()->get_seismic_AllFullnames();
		dataTinyname = manager->getManagerWidget()->get_seismic_AllTinynames();
	}
	else if ( manager->getManagerWidgetV2() )
	{
		manager->getManagerWidgetV2()->seimsicDatabaseUpdate();
		dataFullname = manager->getManagerWidgetV2()->getSeismicAllPath();
		dataTinyname = manager->getManagerWidgetV2()->getSeismicAllNames();
		dataDimx = manager->getManagerWidgetV2()->getSeismicAllDimx();
		dataDimy = manager->getManagerWidgetV2()->getSeismicAllDimy();
		dataDimz = manager->getManagerWidgetV2()->getSeismicAllDimz();
	}

	m_userName = getUserName();

	long N = std::min(dataFullname.size(), dataTinyname.size());
	if ( dimx < 0 && dimy < 0 && dimz < 0 )
	{
		for (long i=0; i<N; i++) {
			QString name = dataTinyname[i];
			// name.replace(".txt", "");
			SeismicInformation* info = new SeismicInformation(name, dataFullname[i], manager, m_enableToggleAction);
			info->setUserName(m_userName);
			insert(i, info);
		}
	}
	else
	{
		for (long i=0; i<N; i++) {
			if ( dataDimx[i] != dimx || dataDimy[i] != dimy || dataDimz[i] != dimz ) continue;
			QString name = dataTinyname[i];
			// name.replace(".txt", "");
			SeismicInformation* info = new SeismicInformation(name, dataFullname[i], manager, m_enableToggleAction);
			info->setUserName(m_userName);
			insert(i, info);
		}
	}
}

SeismicInformationAggregator::~SeismicInformationAggregator() {

}

QString SeismicInformationAggregator::surveyName()
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

QString SeismicInformationAggregator::projectName()
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

bool SeismicInformationAggregator::isCreatable() const {
	return true;
}

bool SeismicInformationAggregator::createStorage() {
	// NurbsWidget::showWidget();
	return true;
}

bool SeismicInformationAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	SeismicInformation* info = dynamic_cast<SeismicInformation*>(information);
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

SeismicInformation* SeismicInformationAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const SeismicInformation* SeismicInformationAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> SeismicInformationAggregator::begin() {
	SeismicInformationIterator* it = new SeismicInformationIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> SeismicInformationAggregator::end() {
	SeismicInformationIterator* it = new SeismicInformationIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long SeismicInformationAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> SeismicInformationAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE};
}

void SeismicInformationAggregator::insert(long i, SeismicInformation* information) {
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

	emit seismicInformationAdded(i, information);
	emit informationAdded(information);
}

void SeismicInformationAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	SeismicInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	emit seismicInformationRemoved(i, information);
	emit informationRemoved(information);
}

QString SeismicInformationAggregator::getUserName()
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

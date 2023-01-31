#include "loginformation.h"

#include "folderdata.h"
// #include "seismicinformationmetadatawidget.h"
// #include "seismicinformationpanelwidget.h"
// #include "nurbswidget.h"
#include "propertyfiltersparser.h"
#include <freeHorizonQManager.h>
#include <seismicsurvey.h>
#include <seismic3dabstractdataset.h>
#include <DataSelectorDialog.h>
#include <GeotimeProjectManagerWidget.h>

#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPainter>
#include <QPixmap>
#include <QTextStream>
#include <QWidget>
#include <QProcess>
#include <Xt.h>

LogInformation::LogInformation(const QString& name, const QString& fullPath, WorkingSetManager* manager,
		QObject* parent) : IInformation(parent), m_name(name), m_fullPath(fullPath), m_manager(manager) {

}

LogInformation::~LogInformation() {

}

// for actions
bool LogInformation::isDeletable() const {
	return true;
}

bool LogInformation::deleteStorage(QString* errorMsg) {
	/*
	if (isSelected()) {
		toggleSelection(false);
	}

	QFileInfo fileInfo(m_fullPath);
	QString fileOwner = fileInfo.owner();
	if ( fileOwner != m_userName ) {
		if (errorMsg) {
			*errorMsg = tr("Current user (")+m_userName+tr(") is not the owner (")+fileOwner+tr(") of the dataset");
		}
		return false;
	}
	QFile file(m_fullPath);
	bool success = file.remove();
	if (errorMsg && !success) {
		*errorMsg = tr("Failed to delete : ") + m_fullPath;
	} else if (errorMsg) {
		*errorMsg = "";
	}
	return success;
	*/
	return true;
}

bool LogInformation::isSelectable() const {
	// return !m_manager.isNull();
	return true;
}

bool LogInformation::isSelected() const {
	/*
	if (!isSelectable()) {
		return false;
	}
	QList<IData*> datas = m_manager->folders().seismics->data();
	bool notFound = true;

	for (int j=0; j<datas.size(); j++)
	{
		SeismicSurvey *survey = dynamic_cast<SeismicSurvey*>(datas[j]);
		if ( survey )
		{
			QList<Seismic3DAbstractDataset*> dataSets = survey->datasets();
			long i = 0;
			while (notFound && i < dataSets.size()) {
				Seismic3DAbstractDataset *dataSet = dataSets[i];
				QString name = dataSet->name();
				notFound = name.compare(m_name) != 0;
				i++;
			}
		}
	}
	return !notFound;
	*/
	return false;
}

void LogInformation::toggleSelection(bool toggle) {
	/*
	if (!isSelectable()) {
		return;
	}

	bool currentState = isSelected();
	if (currentState==toggle) {
		return;
	}

	bool toggleValid = false;
	if ( !m_manager->getManagerWidget() ) return;
	if ( toggle )
	{
		m_manager->getManagerWidget()->set_seismics_filter("");

		m_manager->getManagerWidget()->clear_seismic_gui_selection();
		bool valid = m_manager->getManagerWidget()->select_seismic_tinyname(m_name);

		m_manager->getManagerWidget()->trt_seismic_basket_add();

		bool bIsNewSurvey = false;

		QString surveyName = m_manager->getManagerWidget()->get_survey_name();
		QString surveyPath = m_manager->getManagerWidget()->get_survey_fullpath_name();


		SeismicSurvey* baseSurvey = DataSelectorDialog::dataGetBaseSurvey(m_manager.data(), surveyName, surveyPath, bIsNewSurvey);
		if(baseSurvey != nullptr){
			std::vector<QString> datasetNames = m_manager->getManagerWidget()->get_seismic_names();
			std::vector<QString> datasetPaths = m_manager->getManagerWidget()->get_seismic_fullpath_names();
			DataSelectorDialog::createSeismic(baseSurvey,m_manager.data(),datasetPaths,datasetNames,bIsNewSurvey);
			toggleValid = true;
		}
	}
	else
	{
		m_manager->getManagerWidget()->set_seismics_filter("");

		m_manager->getManagerWidget()->clear_seismic_basket_gui_selection();
		bool valid = m_manager->getManagerWidget()->select_seismic_basket_tinyname(m_name);

		m_manager->getManagerWidget()->trt_seismic_basket_sub();

		bool bIsNewSurvey = false;

		QString surveyName = m_manager->getManagerWidget()->get_survey_name();
		QString surveyPath = m_manager->getManagerWidget()->get_survey_fullpath_name();

		SeismicSurvey* baseSurvey = DataSelectorDialog::dataGetBaseSurvey(m_manager.data(), surveyName, surveyPath, bIsNewSurvey);
		if(baseSurvey != nullptr){
			std::vector<QString> datasetNames = m_manager->getManagerWidget()->get_seismic_names();
			std::vector<QString> datasetPaths = m_manager->getManagerWidget()->get_seismic_fullpath_names();
			DataSelectorDialog::createSeismic(baseSurvey,m_manager.data(),datasetPaths,datasetNames,bIsNewSurvey);
			toggleValid = true;
		}
	}

	if (toggleValid && m_manager && m_manager->getManagerWidget()) {
		m_manager->getManagerWidget()->save_to_default_session();
	}
	*/

	/*
	QString nurbsName = m_name;
	nurbsName.replace(".txt", "");
	if (toggle) {
		nurbsName += ".txt";
		// select data
		NurbsWidget::addNurbs(m_fullPath, nurbsName);
	} else {
		// unselect data
		NurbsWidget::removeNurbs(m_fullPath, nurbsName);
	}
	*/
}

// comments
bool LogInformation::commentsEditable() const {
	return false;
}

QString LogInformation::comments() const {
	return "";
}

void LogInformation::setComments(const QString& txt) {
	// no comments
}

bool LogInformation::hasIcon() const {
	return false;
}

QIcon LogInformation::icon(int preferedSizeX, int preferedSizeY) const {
	// QImage img(preferedSizeX, preferedSizeY, QImage::Format_RGB32);
	// QPainter p(&img);
	// p.fillRect(img.rect(), m_color);
	// QPixmap pixmap = QPixmap::fromImage(img);
	// return QIcon(pixmap);
	return QIcon();
}

// for sort and filtering
QString LogInformation::mainOwner() const {
	QString txt;
	if (m_cacheOwners.size()>0) {
		txt = m_cacheOwners[0];
	} else {
		QStringList ownerList = owners();
		if (ownerList.size()>0) {
			txt = ownerList[0];
		}
	}
	return txt;
}

QStringList LogInformation::owners() const {
	if (m_cacheOwners.size()==0) {
		searchFileCache();
	}
	return m_cacheOwners;
}

QDateTime LogInformation::mainCreationDate() const {
	QDateTime date;
	if (m_cacheCreationDates.size()>0) {
		date = m_cacheCreationDates[0];
	} else {
		QList<QDateTime> dateList = creationDates();
		if (dateList.size()>0) {
			date = dateList[0];
		}
	}
	return date;
}

QList<QDateTime> LogInformation::creationDates() const {
	if (m_cacheCreationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheCreationDates;
}

QDateTime LogInformation::mainModificationDate() const {
	QDateTime date;
	if (m_cacheModificationDates.size()>0) {
		date = m_cacheModificationDates[0];
	} else {
		QList<QDateTime> dateList = modificationDates();
		if (dateList.size()>0) {
			date = dateList[0];
		}
	}
	return date;
}

QList<QDateTime> LogInformation::modificationDates() const {
	if (m_cacheModificationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheModificationDates;
}

QString LogInformation::name() const {
	return m_name;
}

information::StorageType LogInformation::storage() const {
	return information::StorageType::NEXTVISION;
}

void LogInformation::searchFileCache() const {
	if (m_cacheSearchDone) {
		return;
	}

	m_cacheCreationDates.clear();
	m_cacheModificationDates.clear();
	m_cacheOwners.clear();

	QFileInfo txtFile(m_fullPath);
	if (txtFile.exists()) {
		QDateTime mainCreationTime = txtFile.birthTime();
		QDateTime mainModificationTime = txtFile.lastModified();
		QString mainOwner = txtFile.owner();
		m_cacheCreationDates.append(mainCreationTime);
		m_cacheModificationDates.append(mainModificationTime);
		m_cacheOwners.append(mainOwner);
	}

	QFileInfo objFile(txtFile.dir().absoluteFilePath(txtFile.baseName()+".obj"));
	if (objFile.exists()) {
		QDateTime creationTime = objFile.birthTime();
		QDateTime modificationTime = objFile.lastModified();
		QString owner = objFile.owner();
		if (!m_cacheCreationDates.contains(creationTime)) {
			m_cacheCreationDates.append(creationTime);
		}
		if (!m_cacheModificationDates.contains(modificationTime)) {
			m_cacheModificationDates.append(modificationTime);
		}
		if (!m_cacheOwners.contains(owner)) {
			m_cacheOwners.append(owner);
		}
	}
}

bool LogInformation::hasProperty(information::Property property) const {
	return property==information::Property::CREATION_DATE || property==information::Property::MODIFICATION_DATE ||
			property==information::Property::NAME || property==information::Property::OWNER ||
			property==information::Property::STORAGE_TYPE;
}

QVariant LogInformation::property(information::Property property) const {
	QVariant out;

	switch (property) {
	case information::Property::CREATION_DATE:
		out = PropertyFiltersParser::toVariant(creationDates());
		break;
	case information::Property::MODIFICATION_DATE:
		out = PropertyFiltersParser::toVariant(modificationDates());
		break;
	case information::Property::NAME:
		out = name();
		break;
	case information::Property::OWNER:
		out = owners();
		break;
	case information::Property::STORAGE_TYPE:
		out = PropertyFiltersParser::storageToString(storage());
		break;
	}

	return out;
}

bool LogInformation::isCompatible(information::Property prop, const QVariant& filter) const {
	if (!hasProperty(prop)) {
		return false;
	}

	QVariant value = this->property(prop);
	return PropertyFiltersParser::isCompatible(prop, filter, value);
}

QColor LogInformation::color() const {
	return m_color;
}

void LogInformation::setColor(QColor color) {
//	if (m_color!=color) {
//		m_color = color;
//		if (isSelected()) {
//			NurbsWidget::setColor(m_name, m_color);
//		}
//		NurbsWidget::saveColor(m_fullPath, m_color);
//		emit colorChanged(m_color);
//		emit iconChanged();
//	}
}

QString LogInformation::getSize()
{
	inri::Xt xt((char*)m_fullPath.toStdString().c_str());
	if ( !xt.is_valid() ) return "";
	int dimx = xt.nSamples();
	int dimy = xt.nRecords();
	int dimz = xt.nSlices();
	QString ret = QString::number(dimx) + " x " + QString::number(dimy) + " x " + QString::number(dimz);
	return ret;
}

QString LogInformation::getSizeOnDisk()
{
	FILE* f = fopen((char*)m_fullPath.toStdString().c_str(), "r");
	std::size_t size = 0;
	if (f!=nullptr) {
		fseek(f, 0L, SEEK_END);
		size = ftell(f);
		fclose(f);
	}

	if ( size < 1000 ) return QString::number(size) + " bytes";
	if ( size < 1000000 ) return QString::number((double)size/1000.0) + " kbytes";
	if ( size < 1000000000) return QString::number((double)size/1000000.0) + " Mbytes";
	return QString::number((double)size/1000000000.0) + " Gbytes";
}

QString LogInformation::getVoxelFormat()
{
	inri::Xt xt((char*)m_fullPath.toStdString().c_str());
	if ( !xt.is_valid() ) return "";
	inri::Xt::Type type = xt.type();
	QString ret = QString::fromStdString(inri::Xt::type2str(type));
	return ret;
}

QString LogInformation::getAxis()
{
	inri::Xt xt((char*)m_fullPath.toStdString().c_str());
	if ( !xt.is_valid() ) return "";
	inri::Xt::Axis axis = xt.axis();
	return QString::fromStdString(inri::Xt::axis2str(axis));
}

QString LogInformation::getDataSetType()
{
	if ( m_name.contains("nextvisionpatch") ) return "patch";
	if ( m_name.contains("rgt") ) return "RGT";
	if ( m_name.contains("dipxy") ) return "dip xy";
	if ( m_name.contains("dipxz") ) return "dip xz";
	return "seismic";
}

std::vector<QString> LogInformation::getDataParams()
{
	std::vector<QString> ret;
	inri::Xt xt((char*)m_fullPath.toStdString().c_str());
	if ( !xt.is_valid() ) return ret;
	float stepSlices = xt.stepSlices();
	float stepRecords = xt.stepRecords();
	float stepSamples = xt.stepSamples();
	float startSlice = xt.startSlice();
	float startRecord = xt.startRecord();
	float startSamples = xt.startSamples();
	ret.push_back(QString::number(startSamples));
	ret.push_back(QString::number(stepSamples));
	ret.push_back(QString::number(startRecord));
	ret.push_back(QString::number(stepRecords));
	ret.push_back(QString::number(startSlice));
	ret.push_back(QString::number(stepSlices));
	return ret;
}

std::vector<float> LogInformation::getDataFloatParams()
{
	std::vector<float> ret;
	inri::Xt xt((char*)m_fullPath.toStdString().c_str());
	if ( !xt.is_valid() ) return ret;
	float stepSlices = xt.stepSlices();
	float stepRecords = xt.stepRecords();
	float stepSamples = xt.stepSamples();
	float startSlice = xt.startSlice();
	float startRecord = xt.startRecord();
	float startSamples = xt.startSamples();
	ret.push_back(startSamples);
	ret.push_back(stepSamples);
	ret.push_back(startRecord);
	ret.push_back(stepRecords);
	ret.push_back(startSlice);
	ret.push_back(stepSlices);
	return ret;
}

std::vector<int> LogInformation::getDims()
{
	std::vector<int> ret;
	inri::Xt xt((char*)m_fullPath.toStdString().c_str());
	if ( !xt.is_valid() ) return ret;
	int dimx = xt.nSamples();
	int dimy = xt.nRecords();
	int dimz = xt.nSlices();
	ret.push_back(dimx);
	ret.push_back(dimy);
	ret.push_back(dimz);
	return ret;
}




// for gui representation, maybe should be done by another class
IInformationPanelWidget* LogInformation::buildInformationWidget(QWidget* parent) {
	// return new LogInformationPanelWidget(this, parent);
	return nullptr;
}

QWidget* LogInformation::buildMetadataWidget(QWidget* parent) {
	// return new LogInformationMetadataWidget(this, parent);
	return nullptr;
}

QString LogInformation::folder() const {
	return QFileInfo(m_fullPath).absolutePath();
}

QString LogInformation::mainPath() const {
	return m_fullPath;
}

void LogInformation::setUserName(QString name)
{
	m_userName = name;
	m_userName.replace("\n", "");
}

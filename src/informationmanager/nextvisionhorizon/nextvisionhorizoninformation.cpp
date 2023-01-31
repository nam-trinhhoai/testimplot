#include "nextvisionhorizoninformation.h"

#include "DataSelectorDialog.h"
#include "folderdata.h"
#include "freehorizon.h"
#include "GeotimeProjectManagerWidget.h"
#include "nextvisionhorizoninformationmetadatawidget.h"
#include "nextvisionhorizoninformationpanelwidget.h"
// #include "nurbswidget.h"
#include "propertyfiltersparser.h"
#include <freeHorizonQManager.h>
#include <freeHorizonManager.h>

#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPainter>
#include <QPixmap>
#include <QTextStream>
#include <QWidget>
#include <Xt.h>

NextvisionHorizonInformation::NextvisionHorizonInformation(const QString& name, const QString& fullPath, WorkingSetManager* manager, bool enableToggleAction,
		QObject* parent) : IInformation(parent), m_name(name), m_fullPath(fullPath), m_manager(manager) {

	// read txt file
	bool ok = true;
	m_enableToggleAction = enableToggleAction;
	m_color = FreeHorizonQManager::loadColorFromPath(m_fullPath, &ok);
	m_attributName = FreeHorizonQManager::getAttributData(m_fullPath);
	m_attributPath = FreeHorizonQManager::getAttributPath(m_fullPath);

}

NextvisionHorizonInformation::~NextvisionHorizonInformation() {

}

// for actions
bool NextvisionHorizonInformation::isDeletable() const {
	return true;
}

bool NextvisionHorizonInformation::deleteStorage(QString* errorMsgPtr) {
	if (isSelected()) {
		toggleSelection(false);
	}

	std::string errMsg;
	bool success = FreeHorizonManager::erase(m_fullPath.toStdString(), &errMsg);

	if (!success) {
		qDebug() << "Failed to delete horizon : " << QString::fromStdString(errMsg);
		if (errorMsgPtr) {
			*errorMsgPtr = tr("Failed to delete horizon : ") + QString::fromStdString(errMsg);
		}
	} else if (errorMsgPtr) {
		*errorMsgPtr = "";
	}

	return success;
}

bool NextvisionHorizonInformation::isSelectable() const {
	return !m_manager.isNull();
}

bool NextvisionHorizonInformation::isSelected() const {
	if (!isSelectable()) {
		return false;
	}
	if ( !m_enableToggleAction ) return false;

	searchLoadedData();

	return m_loadedData!=nullptr;
}

void NextvisionHorizonInformation::toggleSelection(bool toggle) {
	if (!isSelectable()) {
		return;
	}

	bool currentState = isSelected();
	if (currentState==toggle) {
		return;
	}
	if ( !m_enableToggleAction ) return;
	if (toggle && m_manager && m_manager->getManagerWidget()) {
		QString surveyPath = m_manager->getManagerWidget()->get_survey_fullpath_name();
		QString surveyName = m_manager->getManagerWidget()->get_survey_name();
		bool bIsNewSurvey = false;
		SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(m_manager, surveyName, surveyPath, bIsNewSurvey);

		DataSelectorDialog::addNVHorizons(m_manager, survey, {m_fullPath}, {m_name});

		if (m_manager!=nullptr && m_manager->getManagerWidget()!=nullptr) {
			m_manager->getManagerWidget()->add_freehorizon(m_fullPath, m_name);
			m_manager->getManagerWidget()->save_to_default_session();
		}

		searchLoadedData();
	} else if (!toggle && m_loadedData && m_manager) {
		FreeHorizon* horizon = m_loadedData;
		disconnect(m_loadedData, &FreeHorizon::destroyed, this, &NextvisionHorizonInformation::loadedDataDestroyed);
		disconnect(m_loadedData, &FreeHorizon::colorChanged, this, &NextvisionHorizonInformation::loadedDataColorChanged);
		m_loadedData = nullptr;
		m_manager->removeFreeHorizons(horizon); // this should delete horizon

		if (m_manager!=nullptr && m_manager->getManagerWidget()!=nullptr) {
			m_manager->getManagerWidget()->remove_freehorizon(m_fullPath);
			m_manager->getManagerWidget()->save_to_default_session();
		}
	}
}

// comments
bool NextvisionHorizonInformation::commentsEditable() const {
	return false;
}

QString NextvisionHorizonInformation::comments() const {
	return "";
}

void NextvisionHorizonInformation::setComments(const QString& txt) {
	// no comments
}

bool NextvisionHorizonInformation::hasIcon() const {
	return true;
}

QIcon NextvisionHorizonInformation::icon(int preferedSizeX, int preferedSizeY) const {
	// QImage img(preferedSizeX, preferedSizeY, QImage::Format_RGB32);
	// QPainter p(&img);
	// p.fillRect(img.rect(), m_color);
	// QPixmap pixmap = QPixmap::fromImage(img);
	// return QIcon(pixmap);
	return FreeHorizonQManager::getHorizonIcon(m_fullPath);
}

// for sort and filtering
QString NextvisionHorizonInformation::mainOwner() const {
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

QStringList NextvisionHorizonInformation::owners() const {
	if (m_cacheOwners.size()==0) {
		searchFileCache();
	}
	return m_cacheOwners;
}

QDateTime NextvisionHorizonInformation::mainCreationDate() const {
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

QList<QDateTime> NextvisionHorizonInformation::creationDates() const {
	if (m_cacheCreationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheCreationDates;
}

QDateTime NextvisionHorizonInformation::mainModificationDate() const {
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

QList<QDateTime> NextvisionHorizonInformation::modificationDates() const {
	if (m_cacheModificationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheModificationDates;
}

QString NextvisionHorizonInformation::name() const {
	return m_name;
}

QString NextvisionHorizonInformation::path() const {
	return m_fullPath;
}

information::StorageType NextvisionHorizonInformation::storage() const {
	return information::StorageType::NEXTVISION;
}

void NextvisionHorizonInformation::searchFileCache() const {
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

bool NextvisionHorizonInformation::hasProperty(information::Property property) const {
	return property==information::Property::CREATION_DATE || property==information::Property::MODIFICATION_DATE ||
			property==information::Property::NAME || property==information::Property::OWNER ||
			property==information::Property::STORAGE_TYPE;
}

QVariant NextvisionHorizonInformation::property(information::Property property) const {
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

bool NextvisionHorizonInformation::isCompatible(information::Property prop, const QVariant& filter) const {
	if (!hasProperty(prop)) {
		return false;
	}

	QVariant value = this->property(prop);
	return PropertyFiltersParser::isCompatible(prop, filter, value);
}

QColor NextvisionHorizonInformation::color() const {
	return m_color;
}

void NextvisionHorizonInformation::setColor(QColor color) {
	if (m_color!=color) {
		m_color = color;
		FreeHorizonQManager::saveColorToPath(m_fullPath, m_color);
		if (isSelected() && m_loadedData && m_color!=m_loadedData->color()) {
			m_loadedData->setColor(m_color);
		}
		emit colorChanged(m_color);
		emit iconChanged();
	}
}

QString NextvisionHorizonInformation::getVoxelFormat()
{
	inri::Xt xt((char*)m_fullPath.toStdString().c_str());
	if ( !xt.is_valid() ) return "";
	inri::Xt::Type type = xt.type();
	QString ret = QString::fromStdString(inri::Xt::type2str(type));
	return ret;
}

QString NextvisionHorizonInformation::getAxis()
{
	inri::Xt xt((char*)m_fullPath.toStdString().c_str());
	if ( !xt.is_valid() ) return "";
	inri::Xt::Axis axis = xt.axis();
	return QString::fromStdString(inri::Xt::axis2str(axis));
}

QString NextvisionHorizonInformation::getDataSetType()
{
	if ( m_name.contains("nextvisionpatch") ) return "patch";
	if ( m_name.contains("rgt") ) return "RGT";
	if ( m_name.contains("dipxy") ) return "dip xy";
	if ( m_name.contains("dipxz") ) return "dip xz";
	return "seismic";
}

std::vector<QString> NextvisionHorizonInformation::getDataParams()
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


// for gui representation, maybe should be done by another class
IInformationPanelWidget* NextvisionHorizonInformation::buildInformationWidget(QWidget* parent) {
	return new NextvisionHorizonInformationPanelWidget(this, m_manager, parent);
}

QWidget* NextvisionHorizonInformation::buildMetadataWidget(QWidget* parent) {
	return new NextvisionHorizonInformationMetadataWidget(this, parent);
}

QString NextvisionHorizonInformation::folder() const {
	return QFileInfo(m_fullPath).absolutePath();
}

QString NextvisionHorizonInformation::mainPath() const {
	return m_fullPath;
}

std::vector<QString> NextvisionHorizonInformation::attributName()
{
	return m_attributName;
}

std::vector<QString> NextvisionHorizonInformation::attributPath()
{
	return m_attributPath;
}

QString NextvisionHorizonInformation::Dims()
{
	int dimy = 0;
	int dimz = 0;
	FreeHorizonManager::getHorizonDims(m_fullPath.toStdString(), &dimy, &dimz);
	return QString::number(dimy) + " x "  + QString::number(dimz);
}


QString NextvisionHorizonInformation::getNbreAttributs()
{
	int cpt = 0;
	for (int i=0; i<m_attributName.size(); i++)
	{
		QFileInfo f(m_attributName[i]);
		QString ext = f.completeSuffix();
		if ( ext == "raw" ) cpt++;
	}
	return QString::number(cpt);
}


QString NextvisionHorizonInformation::getAttributType(int i)
{
	if ( i < 0 || i > m_attributPath.size() ) return "";
	if ( m_attributName[i] == "isochrone.iso" ) return "isochrone";
	return FreeHorizonQManager::getPrefixFromFile(m_attributName[i]);
}

QString NextvisionHorizonInformation::getSizeOnDisk(int i)
{
	if ( i < 0 || i > m_attributPath.size() ) return "";
	FILE* f = fopen((char*)m_attributPath[i].toStdString().c_str(), "r");
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


QString NextvisionHorizonInformation::getNbreSpectrumFrequencies(int i)
{
	if ( i < 0 || i > m_attributPath.size() ) return "";
	if ( getAttributType(i) != "spectrum" ) return "";
	int nfreq = FreeHorizonManager::getNbreSpectrumFreq(m_attributPath[i].toStdString());
	return QString::number(nfreq);
}

QString NextvisionHorizonInformation::getNbreGccScales(int i)
{
	if ( i < 0 || i > m_attributPath.size() ) return "";
	if ( getAttributType(i) != "gcc" ) return "";
	int nscales = FreeHorizonManager::getNbreGccScales(m_attributPath[i].toStdString());
	return QString::number(nscales);
}

void NextvisionHorizonInformation::loadedDataColorChanged(QColor color) {
	setColor(color);
}

void NextvisionHorizonInformation::loadedDataDestroyed(QObject* obj) {
	if (m_loadedData==obj) {
		m_loadedData = nullptr;
	}
}

void NextvisionHorizonInformation::searchLoadedData() const {
	if (m_loadedData || m_manager==nullptr) {
		return;
	}

	QList<IData*> datas = m_manager->folders().horizonsFree->data();
	long i=0;
	bool notFound = true;
	while (notFound && i<datas.size()) {
		FreeHorizon* horizon = dynamic_cast<FreeHorizon*>(datas[i]);
		notFound = horizon!=nullptr && datas[i]->name().compare(m_name)!=0;
		if (notFound) {
			i++;
		}
	}

	if (!notFound && i<datas.size()) {
		m_loadedData = dynamic_cast<FreeHorizon*>(datas[i]);
		if (m_loadedData) {
			connect(m_loadedData, &FreeHorizon::destroyed, this, &NextvisionHorizonInformation::loadedDataDestroyed);
			connect(m_loadedData, &FreeHorizon::colorChanged, this, &NextvisionHorizonInformation::loadedDataColorChanged);
		}
	}
}




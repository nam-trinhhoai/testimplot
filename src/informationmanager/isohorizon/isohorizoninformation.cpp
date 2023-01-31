#include "isohorizoninformation.h"

#include "DataSelectorDialog.h"
#include "folderdata.h"
#include "GeotimeProjectManagerWidget.h"
#include "isohorizon.h"
#include "isohorizoninformationmetadatawidget.h"
#include "isohorizoninformationpanelwidget.h"
// #include "nurbswidget.h"
#include "propertyfiltersparser.h"
#include <freeHorizonQManager.h>
#include <freeHorizonManager.h>
#include "isoHorizonManager.h"

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

IsoHorizonInformation::IsoHorizonInformation(const QString& name, const QString& fullPath, WorkingSetManager* manager,
		QObject* parent) : IInformation(parent), m_name(name), m_fullPath(fullPath), m_manager(manager) {

	// read txt file
	bool ok = true;
	m_fullPath00000 = m_fullPath + "/iso_00000";
	m_color = FreeHorizonQManager::loadColorFromPath(m_fullPath, &ok);
	m_attributName = FreeHorizonQManager::getAttributData(m_fullPath00000);
	m_attributPath = FreeHorizonQManager::getAttributPath(m_fullPath00000);

	FreeHorizonManager::PARAM horizonParam = FreeHorizonManager::dataSetGetParam(m_fullPath.toStdString()+"/iso_00000/"+FreeHorizonManager::isoDataName);
	if (horizonParam.axis==inri::Xt::Time) {
		m_sampleUnit = SampleUnit::TIME;
	} else if (horizonParam.axis==inri::Xt::Depth) {
		m_sampleUnit = SampleUnit::DEPTH;
	} else {
		m_sampleUnit = SampleUnit::NONE;
	}
}

IsoHorizonInformation::~IsoHorizonInformation() {

}

// for actions
bool IsoHorizonInformation::isDeletable() const {
	return true;
}

bool IsoHorizonInformation::deleteStorage(QString* errorMsgPtr) {
	if (isSelected()) {
		toggleSelection(false);
	}

	std::string errMsg;
	bool success = IsoHorizonManager::erase(m_fullPath.toStdString(), &errMsg);

	if (!success) {
		qDebug() << "Failed to delete rgt iso : " << QString::fromStdString(errMsg);
		if (errorMsgPtr) {
			*errorMsgPtr = tr("Failed to delete rgt iso : ") + QString::fromStdString(errMsg);
		}
	} else if (errorMsgPtr) {
		*errorMsgPtr = "";
	}

	return success;
}

bool IsoHorizonInformation::isSelectable() const {
	return !m_manager.isNull();
}

bool IsoHorizonInformation::isSelected() const {
	if (!isSelectable()) {
		return false;
	}

	searchLoadedData();

	return m_loadedData!=nullptr;
}

void IsoHorizonInformation::toggleSelection(bool toggle) {
	if (!isSelectable()) {
		return;
	}

	bool currentState = isSelected();
	if (currentState==toggle) {
		return;
	}
	if (toggle && m_manager && m_manager->getManagerWidget()) {
		QString surveyPath = m_manager->getManagerWidget()->get_survey_fullpath_name();
		QString surveyName = m_manager->getManagerWidget()->get_survey_name();
		bool bIsNewSurvey = false;
		SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(m_manager, surveyName, surveyPath, bIsNewSurvey);

		DataSelectorDialog::addNVIsoHorizons(m_manager, survey, {m_fullPath}, {m_name});

		if (m_manager!=nullptr && m_manager->getManagerWidget()!=nullptr) {
			m_manager->getManagerWidget()->add_isohorizon(m_fullPath, m_name);
			m_manager->getManagerWidget()->save_to_default_session();
		}

		searchLoadedData();
	} else if (!toggle && m_loadedData && m_manager) {
		IsoHorizon* horizon = m_loadedData;
		disconnect(m_loadedData, &IsoHorizon::destroyed, this, &IsoHorizonInformation::loadedDataDestroyed);
		disconnect(m_loadedData, &IsoHorizon::colorChanged, this, &IsoHorizonInformation::loadedDataColorChanged);
		m_loadedData = nullptr;
		m_manager->removeIsoHorizons(horizon); // this should delete horizon

		if (m_manager!=nullptr && m_manager->getManagerWidget()!=nullptr) {
			m_manager->getManagerWidget()->remove_isohorizon(m_fullPath);
			m_manager->getManagerWidget()->save_to_default_session();
		}
	}
}

// comments
bool IsoHorizonInformation::commentsEditable() const {
	return false;
}

QString IsoHorizonInformation::comments() const {
	return "";
}

void IsoHorizonInformation::setComments(const QString& txt) {
	// no comments
}

bool IsoHorizonInformation::hasIcon() const {
	return true;
}

QIcon IsoHorizonInformation::icon(int preferedSizeX, int preferedSizeY) const {
	return FreeHorizonQManager::getHorizonIcon(m_color, m_sampleUnit);
}

// for sort and filtering
QString IsoHorizonInformation::mainOwner() const {
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

QStringList IsoHorizonInformation::owners() const {
	if (m_cacheOwners.size()==0) {
		searchFileCache();
	}
	return m_cacheOwners;
}

QDateTime IsoHorizonInformation::mainCreationDate() const {
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

QList<QDateTime> IsoHorizonInformation::creationDates() const {
	if (m_cacheCreationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheCreationDates;
}

QDateTime IsoHorizonInformation::mainModificationDate() const {
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

QList<QDateTime> IsoHorizonInformation::modificationDates() const {
	if (m_cacheModificationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheModificationDates;
}

QString IsoHorizonInformation::name() const {
	return m_name;
}

information::StorageType IsoHorizonInformation::storage() const {
	return information::StorageType::NEXTVISION;
}

void IsoHorizonInformation::searchFileCache() const {
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

bool IsoHorizonInformation::hasProperty(information::Property property) const {
	return property==information::Property::CREATION_DATE || property==information::Property::MODIFICATION_DATE ||
			property==information::Property::NAME || property==information::Property::OWNER ||
			property==information::Property::STORAGE_TYPE;
}

QVariant IsoHorizonInformation::property(information::Property property) const {
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

bool IsoHorizonInformation::isCompatible(information::Property prop, const QVariant& filter) const {
	if (!hasProperty(prop)) {
		return false;
	}

	QVariant value = this->property(prop);
	return PropertyFiltersParser::isCompatible(prop, filter, value);
}

QColor IsoHorizonInformation::color() const {
	return m_color;
}

void IsoHorizonInformation::setColor(QColor color) {
	if (m_color!=color) {
		m_color = color;
		FreeHorizonQManager::saveColorToPath(m_fullPath, m_color);
		if (isSelected() && m_loadedData && color!=m_loadedData->color()) {
			m_loadedData->setColor(m_color);
		}
		emit colorChanged(m_color);
		emit iconChanged();
	}
}


// for gui representation, maybe should be done by another class
IInformationPanelWidget* IsoHorizonInformation::buildInformationWidget(QWidget* parent) {
	return new IsoHorizonInformationPanelWidget(this, parent);
}

QWidget* IsoHorizonInformation::buildMetadataWidget(QWidget* parent) {
	return new IsoHorizonInformationMetadataWidget(this, parent);
}

QString IsoHorizonInformation::folder() const {
	return QFileInfo(m_fullPath).absolutePath();
}

QString IsoHorizonInformation::mainPath() const {
	return m_fullPath;
}

QString IsoHorizonInformation::getNbreDirectories()
{
	int n = FreeHorizonQManager::getListName(m_fullPath).size();
	return QString::number(n);

}

QString IsoHorizonInformation::attributDirPath()
{
	return m_fullPath;
}

std::vector<QString> IsoHorizonInformation::attributName()
{
	return m_attributName;
}

std::vector<QString> IsoHorizonInformation::attributPath()
{
	return m_attributPath;
}

QString IsoHorizonInformation::Dims()
{
	int dimy = 0;
	int dimz = 0;
	FreeHorizonManager::getHorizonDims(m_fullPath00000.toStdString(), &dimy, &dimz);
	return QString::number(dimy) + " x "  + QString::number(dimz);
}


QString IsoHorizonInformation::getNbreAttributs()
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


QString IsoHorizonInformation::getAttributType(int i)
{
	if ( i < 0 || i > m_attributPath.size() ) return "";
	if ( m_attributName[i] == "isochrone.iso" ) return "isochrone";
	return FreeHorizonQManager::getPrefixFromFile(m_attributName[i]);
}

QString IsoHorizonInformation::getSizeOnDisk(int i)
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


QString IsoHorizonInformation::getNbreSpectrumFrequencies(int i)
{
	if ( i < 0 || i > m_attributPath.size() ) return "";
	if ( getAttributType(i) != "spectrum" ) return "";
	int nfreq = FreeHorizonManager::getNbreSpectrumFreq(m_attributPath[i].toStdString());
	return QString::number(nfreq);
}

QString IsoHorizonInformation::getNbreGccScales(int i)
{
	if ( i < 0 || i > m_attributPath.size() ) return "";
	if ( getAttributType(i) != "gcc" ) return "";
	int nscales = FreeHorizonManager::getNbreGccScales(m_attributPath[i].toStdString());
	return QString::number(nscales);
}

void IsoHorizonInformation::loadedDataColorChanged(QColor color) {
	setColor(color);
}

void IsoHorizonInformation::loadedDataDestroyed(QObject* obj) {
	if (m_loadedData==obj) {
		m_loadedData = nullptr;
	}
}

void IsoHorizonInformation::searchLoadedData() const {
	if (m_loadedData || m_manager==nullptr) {
		return;
	}

	QList<IData*> datas = m_manager->folders().horizonsIso->data();
	long i=0;
	bool notFound = true;
	while (notFound && i<datas.size()) {
		IsoHorizon* horizon = dynamic_cast<IsoHorizon*>(datas[i]);
		notFound = horizon!=nullptr && datas[i]->name().compare(m_name)!=0;
		if (notFound) {
			i++;
		}
	}

	if (!notFound && i<datas.size()) {
		m_loadedData = dynamic_cast<IsoHorizon*>(datas[i]);
		if (m_loadedData) {
			connect(m_loadedData, &IsoHorizon::destroyed, this, &IsoHorizonInformation::loadedDataDestroyed);
			connect(m_loadedData, &IsoHorizon::colorChanged, this, &IsoHorizonInformation::loadedDataColorChanged);
		}
	}
}

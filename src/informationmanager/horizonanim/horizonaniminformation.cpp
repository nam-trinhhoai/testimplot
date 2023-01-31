#include "horizonaniminformation.h"

#include "folderdata.h"
#include "GeotimeProjectManagerWidget.h"
#include "horizonanimmetadatawidget.h"
#include "horizonanimpanelwidget.h"
//#include "nurbswidget.h"
#include "propertyfiltersparser.h"
#include "util_filesystem.h"


#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPainter>
#include <QPixmap>
#include <QTextStream>
#include <QWidget>

HorizonAnimInformation::HorizonAnimInformation(const QString& name, const QString& fullPath, WorkingSetManager* manager,
		QObject* parent) : IInformation(parent), m_name(name), m_fullPath(fullPath), m_manager(manager) {

	bool valid;
	m_params = HorizonDataRep::readAnimationHorizon(m_fullPath, &valid);

	m_nameAttribut = m_params.attribut;
	m_listHorizons = m_params.horizons;
	searchData();


}


HorizonAnimInformation::HorizonAnimInformation(const QString& name,const QString& fullPath, std::vector<QString> horizons, WorkingSetManager* manager,
		QObject* parent) : IInformation(parent), m_name(name), m_fullPath(fullPath), m_manager(manager) {


	m_nameAttribut = "";
	m_params.attribut = m_nameAttribut;
	m_params.nbHorizons = horizons.size();

	for(int i=0;i<horizons.size();i++)
	{
		m_listHorizons<<horizons[i];
		m_params.horizons.push_back(horizons[i]);
		m_params.orderIndex.push_back(i);
	}
	searchData();

}

void HorizonAnimInformation::searchData()
{
	QList<IData*> datas = m_manager->folders().horizonsAnim->data();

	for(int i=0;i<datas.size();i++)
	{
		HorizonFolderData* data = dynamic_cast<HorizonFolderData*>(datas[i]);
		if(data != nullptr)
		{
			if(data->name() == m_name)
			{
				m_horizonFolderData =  data;
				return;
			}
		}
	}

}


HorizonAnimInformation::~HorizonAnimInformation() {

}

// for actions
bool HorizonAnimInformation::isDeletable() const {
	return true;
}

bool HorizonAnimInformation::deleteStorage(QString* errorMsg) {
	if (isSelected()) {
		toggleSelection(false);
	}

	QFileInfo txtFileInfo(m_fullPath);
	QDir parentDir = txtFileInfo.dir();
	QFileInfo parentInfo(parentDir.absolutePath());
	if (!parentInfo.exists() || !parentInfo.isWritable()) {
		if (errorMsg) {
			*errorMsg = tr("Permission issue on : ") + parentInfo.absoluteFilePath();
		}
		return false;
	}

	QString horPath = parentDir.absoluteFilePath(txtFileInfo.completeBaseName()+".hor");
	QFileInfo horFileInfo(horPath);

	bool objSuccess = true;
	if (horFileInfo.exists()) {
		objSuccess = QFile::remove(horPath);
	}
	if (!objSuccess) {
		qDebug() << "Failed to delete file : " << horPath;
		if (errorMsg) {
			*errorMsg = tr("Failed to delete file : ") + horPath;
		}
	} else if (errorMsg) {
		*errorMsg = "";
	}

	return objSuccess;// && objSuccess;
}

bool HorizonAnimInformation::isSelectable() const {
	return !m_manager.isNull();
}

bool HorizonAnimInformation::isSelected() const {
	if (!isSelectable()) {
		return false;
	}

	QList<IData*> datas = m_manager->folders().horizonsAnim->data();
	long i=0;
	bool notFound = true;
	while (notFound && i<datas.size()) {
		notFound = datas[i]->name().compare(m_name)!=0;
		i++;
	}
	return !notFound;
}

void HorizonAnimInformation::toggleSelection(bool toggle) {
	if (!isSelectable()) {
		return;
	}

	bool currentState = isSelected();
	if (currentState==toggle) {
		return;
	}
	QString horName = m_name;
	horName.replace(".hor", "");
	if (toggle) {
		//horName += ".hor";
		// select data

		 HorizonDataRep::addAnimationHorizon(m_fullPath, horName,m_manager);
		 searchData();

		 if(m_horizonFolderData== nullptr)
			 m_horizonFolderData = new HorizonFolderData(m_manager,horName, m_listHorizons);

			 m_manager->addHorizonAnimData(m_horizonFolderData);
			 m_horizonFolderData->setDisplayPreferences({InlineView,XLineView,RandomView},true);

		if (m_manager!=nullptr && m_manager->getManagerWidget()!=nullptr) {
			m_manager->getManagerWidget()->add_horizonanim(m_fullPath, m_name);
			m_manager->getManagerWidget()->save_to_default_session();
		}
		//NurbsWidget::addNurbs(m_fullPath, nurbsName);
	} else {
		// unselect data
		//NurbsWidget::removeNurbs(m_fullPath, nurbsName);
		//horName += ".hor";
		if(m_horizonFolderData != nullptr) m_manager->removeHorizonAnimData(m_horizonFolderData);
		m_horizonFolderData = nullptr;

		//HorizonDataRep::removeAnimationHorizon(m_fullPath, horName,m_manager);
		if (m_manager!=nullptr && m_manager->getManagerWidget()!=nullptr) {
			m_manager->getManagerWidget()->remove_horizonanim(m_fullPath);
			m_manager->getManagerWidget()->save_to_default_session();
		}
	}
}

// comments
bool HorizonAnimInformation::commentsEditable() const {
	return false;
}

QString HorizonAnimInformation::comments() const {
	return "";
}

void HorizonAnimInformation::setComments(const QString& txt) {
	// no comments
}

bool HorizonAnimInformation::hasIcon() const {
	return false;
}

QIcon HorizonAnimInformation::icon(int preferedSizeX, int preferedSizeY) const {
	QImage img(preferedSizeX, preferedSizeY, QImage::Format_RGB32);
	QPainter p(&img);
	//p.fillRect(img.rect(), m_color);

	QPixmap pixmap = QPixmap::fromImage(img);
	return QIcon(pixmap);
}

// for sort and filtering
QString HorizonAnimInformation::mainOwner() const {
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

QStringList HorizonAnimInformation::owners() const {
	if (m_cacheOwners.size()==0) {
		searchFileCache();
	}
	return m_cacheOwners;
}

QDateTime HorizonAnimInformation::mainCreationDate() const {
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

QList<QDateTime> HorizonAnimInformation::creationDates() const {
	if (m_cacheCreationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheCreationDates;
}

QDateTime HorizonAnimInformation::mainModificationDate() const {
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

QList<QDateTime> HorizonAnimInformation::modificationDates() const {
	if (m_cacheModificationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheModificationDates;
}

QString HorizonAnimInformation::name() const {
	return m_name;
}

information::StorageType HorizonAnimInformation::storage() const {
	return information::StorageType::NEXTVISION;
}

void HorizonAnimInformation::searchFileCache() const {
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

	QFileInfo objFile(txtFile.dir().absoluteFilePath(txtFile.baseName()+".hor"));
	if (objFile.exists()) {
		QDateTime creationTime = objFile.birthTime();
		QDateTime modificationTime = txtFile.lastModified();
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

bool HorizonAnimInformation::hasProperty(information::Property property) const {
	return property==information::Property::CREATION_DATE || property==information::Property::MODIFICATION_DATE ||
			property==information::Property::NAME || property==information::Property::OWNER ||
			property==information::Property::STORAGE_TYPE;
}

QVariant HorizonAnimInformation::property(information::Property property) const {
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

bool HorizonAnimInformation::isCompatible(information::Property prop, const QVariant& filter) const {
	if (!hasProperty(prop)) {
		return false;
	}

	QVariant value = this->property(prop);
	return PropertyFiltersParser::isCompatible(prop, filter, value);
}

QString HorizonAnimInformation::nameAttribut()
{
	return m_nameAttribut;
}

QStringList HorizonAnimInformation::listHorizons()
{
	return m_listHorizons;
}


void HorizonAnimInformation::save() {
	QDir dir(QFileInfo(m_fullPath).absolutePath());
	if(!dir.exists()) mkpath(dir.absolutePath());
	HorizonDataRep::writeAnimationHorizon(m_fullPath, m_params);
}


void HorizonAnimInformation::setNameAttribut(QString name)
{
	m_params.attribut = name;
	m_nameAttribut = name;
}

QString HorizonAnimInformation::folder() const {
	return QFileInfo(m_fullPath).absolutePath();
}

QString HorizonAnimInformation::mainPath() const {
	return m_fullPath;
}


/*

QColor HorizonAnimInformation::color() const {
	return m_color;
}

void HorizonAnimInformation::setColor(QColor color) {
	if (m_color!=color) {
		m_color = color;
		if (isSelected()) {
			NurbsWidget::setColor(m_name, m_color);
		}
		NurbsWidget::saveColor(m_fullPath, m_color);
		emit colorChanged(m_color);
	}
}

int HorizonAnimInformation::nbCurves() const {
	return m_nbCurves;
}

int HorizonAnimInformation::precision() const {
	return m_precision;
}
*/
// for gui representation, maybe should be done by another class
IInformationPanelWidget* HorizonAnimInformation::buildInformationWidget(QWidget* parent) {
	return new HorizonAnimPanelWidget(this, parent);
}

QWidget* HorizonAnimInformation::buildMetadataWidget(QWidget* parent) {
	return new HorizonAnimMetadataWidget(this, parent);
}

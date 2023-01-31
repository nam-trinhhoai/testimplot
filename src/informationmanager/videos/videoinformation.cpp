#include "videoinformation.h"

#include "nurbinformationmetadatawidget.h"
#include "videoinformationpanelwidget.h"
#include "propertyfiltersparser.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>


VideoInformation::VideoInformation(const QString& name, const QString& aviPath,
		QObject* parent) : IInformation(parent), m_name(name), m_aviPath(aviPath) {

}

VideoInformation::~VideoInformation() {

}

// for actions
bool VideoInformation::isDeletable() const {
	return true;
}

bool VideoInformation::deleteStorage(QString* errorMsg) {
	QFileInfo fileInfo(m_aviPath);
	QDir parentDir = fileInfo.dir();
	QFileInfo parentInfo(parentDir.absolutePath());
	if (!parentInfo.exists() || !parentInfo.isWritable()) {
		if (errorMsg!=nullptr) {
			*errorMsg = "Permission issue on : " + parentDir.absolutePath();
		}
		return false;
	}

	bool success = true;
	if (fileInfo.exists()) {
		success = QFile::remove(m_aviPath);
	}

	if (!success) {
		qDebug() << "Failed to delete file : " << m_aviPath;
		if (errorMsg!=nullptr) {
			*errorMsg = "Failed to delete file : " + m_aviPath;
		}
	}

	return success;
}

bool VideoInformation::isSelectable() const {
	return false;
}

bool VideoInformation::isSelected() const {
	return false;
}

void VideoInformation::toggleSelection(bool toggle) {
	// Nothing to do
}

// comments
bool VideoInformation::commentsEditable() const {
	return false;
}

QString VideoInformation::comments() const {
	return "";
}

void VideoInformation::setComments(const QString& txt) {
	// no comments
}

bool VideoInformation::hasIcon() const {
	return false;
}

QIcon VideoInformation::icon(int preferedSizeX, int preferedSizeY) const {
	return QIcon();
}

// for sort and filtering
QString VideoInformation::mainOwner() const {
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

QStringList VideoInformation::owners() const {
	if (m_cacheOwners.size()==0) {
		searchFileCache();
	}
	return m_cacheOwners;
}

QDateTime VideoInformation::mainCreationDate() const {
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

QList<QDateTime> VideoInformation::creationDates() const {
	if (m_cacheCreationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheCreationDates;
}

QDateTime VideoInformation::mainModificationDate() const {
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

QList<QDateTime> VideoInformation::modificationDates() const {
	if (m_cacheModificationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheModificationDates;
}

QString VideoInformation::name() const {
	return m_name;
}

information::StorageType VideoInformation::storage() const {
	return information::StorageType::NEXTVISION;
}

void VideoInformation::searchFileCache() const {
	if (m_cacheSearchDone) {
		return;
	}

	m_cacheCreationDates.clear();
	m_cacheModificationDates.clear();
	m_cacheOwners.clear();

	QFileInfo txtFile(m_aviPath);
	if (txtFile.exists()) {
		QDateTime mainCreationTime = txtFile.birthTime();
		QDateTime mainModificationTime = txtFile.lastModified();
		QString mainOwner = txtFile.owner();
		m_cacheCreationDates.append(mainCreationTime);
		m_cacheModificationDates.append(mainModificationTime);
		m_cacheOwners.append(mainOwner);
	}
}

bool VideoInformation::hasProperty(information::Property property) const {
	return property==information::Property::CREATION_DATE || property==information::Property::MODIFICATION_DATE ||
			property==information::Property::NAME || property==information::Property::OWNER ||
			property==information::Property::STORAGE_TYPE;
}

QVariant VideoInformation::property(information::Property property) const {
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

bool VideoInformation::isCompatible(information::Property prop, const QVariant& filter) const {
	if (!hasProperty(prop)) {
		return false;
	}

	QVariant value = this->property(prop);
	return PropertyFiltersParser::isCompatible(prop, filter, value);
}

// for gui representation, maybe should be done by another class
IInformationPanelWidget* VideoInformation::buildInformationWidget(QWidget* parent) {
	return new VideoInformationPanelWidget(this, parent);
}

QWidget* VideoInformation::buildMetadataWidget(QWidget* parent) {
	return new NurbInformationMetadataWidget(this, parent);
}

QString VideoInformation::folder() const {
	return QFileInfo(m_aviPath).absolutePath();
}

QString VideoInformation::mainPath() const {
	return m_aviPath;
}

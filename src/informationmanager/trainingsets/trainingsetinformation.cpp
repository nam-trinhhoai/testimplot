#include "trainingsetinformation.h"

#include "nurbinformationmetadatawidget.h"
#include "trainingsetinformationpanelwidget.h"
#include "propertyfiltersparser.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QQueue>
#include <QStack>


TrainingSetInformation::TrainingSetInformation(const QString& name, const QString& trainingSetPath,
		QObject* parent) : IInformation(parent), m_name(name), m_trainingSetPath(trainingSetPath) {

}

TrainingSetInformation::~TrainingSetInformation() {

}

// for actions
bool TrainingSetInformation::isDeletable() const {
	return true;
}

bool TrainingSetInformation::deleteStorage(QString* errorMsg) {
	 // this need to be stack to allow to delete files before parent directories
	QStack<QString> pathToCheck;
	pathToCheck.push(m_trainingSetPath);

	if (errorMsg) {
		*errorMsg = "";
	}

	bool valid = true;
	QFileInfoList itemsToDelete;
	while (valid && pathToCheck.size()>0) {
		QString path = pathToCheck.pop();
		QFileInfo fileInfo(path);
		QDir parentDir = fileInfo.dir();
		QFileInfo parentInfo(parentDir.absolutePath());
		valid = fileInfo.exists() && parentInfo.exists() && parentInfo.isWritable() &&
				((fileInfo.isDir() && fileInfo.isExecutable()) || !fileInfo.isDir());
		if (valid) {
			itemsToDelete.append(fileInfo);
		} else if (errorMsg) {
			*errorMsg = "Permission issue on : " + fileInfo.absoluteFilePath();
		}
		if (valid && fileInfo.isDir()) {
			QFileInfoList fileInfos = QDir(fileInfo.absoluteFilePath()).entryInfoList(QStringList() << "*",
					QDir::Dirs | QDir::Files| QDir::NoDotAndDotDot);
			for (long i=0; i<fileInfos.size(); i++) {
				pathToCheck.push(fileInfos[i].absoluteFilePath());
			}
		}
	}

	if (!valid) {
		return false;
	}

	bool success = true;

	long i = itemsToDelete.size()-1;
	while (success && i>=0) {
		const QFileInfo& fileInfo = itemsToDelete[i];
		if (fileInfo.exists() && fileInfo.isFile()) {
			success = QFile::remove(fileInfo.absoluteFilePath());
		}
		if (fileInfo.exists() && fileInfo.isDir()) {
			QDir parentDir = fileInfo.absoluteDir();
			success = parentDir.rmdir(fileInfo.fileName());
		}
		if (!success && errorMsg) {
			*errorMsg = "Failed to delete file : " + fileInfo.absoluteFilePath();
		}
		i--;
	}

	if (!success) {
		qDebug() << "Failed to delete file : " << m_trainingSetPath;
	}

	return success;
}

bool TrainingSetInformation::isSelectable() const {
	return false;
}

bool TrainingSetInformation::isSelected() const {
	return false;
}

void TrainingSetInformation::toggleSelection(bool toggle) {
	// Nothing to do
}

// comments
bool TrainingSetInformation::commentsEditable() const {
	return false;
}

QString TrainingSetInformation::comments() const {
	return "";
}

void TrainingSetInformation::setComments(const QString& txt) {
	// no comments
}

bool TrainingSetInformation::hasIcon() const {
	return false;
}

QIcon TrainingSetInformation::icon(int preferedSizeX, int preferedSizeY) const {
	return QIcon();
}

// for sort and filtering
QString TrainingSetInformation::mainOwner() const {
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

QStringList TrainingSetInformation::owners() const {
	if (m_cacheOwners.size()==0) {
		searchFileCache();
	}
	return m_cacheOwners;
}

QDateTime TrainingSetInformation::mainCreationDate() const {
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

QList<QDateTime> TrainingSetInformation::creationDates() const {
	if (m_cacheCreationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheCreationDates;
}

QDateTime TrainingSetInformation::mainModificationDate() const {
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

QList<QDateTime> TrainingSetInformation::modificationDates() const {
	if (m_cacheModificationDates.size()==0) {
		searchFileCache();
	}
	return m_cacheModificationDates;
}

QString TrainingSetInformation::name() const {
	return m_name;
}

information::StorageType TrainingSetInformation::storage() const {
	return information::StorageType::NEXTVISION;
}

void TrainingSetInformation::searchFileCache() const {
	if (m_cacheSearchDone) {
		return;
	}

	m_cacheCreationDates.clear();
	m_cacheModificationDates.clear();
	m_cacheOwners.clear();

	 // this need to be queue to have the right mainOwner/mainCreationTime/mainModificationTime
	QQueue<QString> pathToCheck;
	pathToCheck.enqueue(m_trainingSetPath);

	while (pathToCheck.size()>0) {
		QString path = pathToCheck.dequeue();
		QFileInfo fileInfo(path);
		if (fileInfo.exists()) {
			QDateTime creationTime = fileInfo.birthTime();
			QDateTime modificationTime = fileInfo.lastModified();
			QString owner = fileInfo.owner();
			if (!m_cacheCreationDates.contains(creationTime)) {
				m_cacheCreationDates.append(creationTime);
			}
			if (!m_cacheModificationDates.contains(modificationTime)) {
				m_cacheModificationDates.append(modificationTime);
			}
			if (!m_cacheOwners.contains(owner)) {
				m_cacheOwners.append(owner);
			}

			if (fileInfo.isDir()) {
				QFileInfoList fileInfos = QDir(fileInfo.absoluteFilePath()).entryInfoList(QStringList() << "*",
						QDir::Dirs | QDir::Files| QDir::NoDotAndDotDot);
				for (long i=0; i<fileInfos.size(); i++) {
					pathToCheck.enqueue(fileInfos[i].absoluteFilePath());
				}
			}
		}
	}
}

bool TrainingSetInformation::hasProperty(information::Property property) const {
	return property==information::Property::CREATION_DATE || property==information::Property::MODIFICATION_DATE ||
			property==information::Property::NAME || property==information::Property::OWNER ||
			property==information::Property::STORAGE_TYPE;
}

QVariant TrainingSetInformation::property(information::Property property) const {
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

bool TrainingSetInformation::isCompatible(information::Property prop, const QVariant& filter) const {
	if (!hasProperty(prop)) {
		return false;
	}

	QVariant value = this->property(prop);
	return PropertyFiltersParser::isCompatible(prop, filter, value);
}

// for gui representation, maybe should be done by another class
IInformationPanelWidget* TrainingSetInformation::buildInformationWidget(QWidget* parent) {
	return new TrainingSetInformationPanelWidget(this, parent);
}

QWidget* TrainingSetInformation::buildMetadataWidget(QWidget* parent) {
	return new NurbInformationMetadataWidget(this, parent);
}

QString TrainingSetInformation::folder() const {
	return m_trainingSetPath;
}

QString TrainingSetInformation::mainPath() const {
	return m_trainingSetPath + "/trainingset.json";
}

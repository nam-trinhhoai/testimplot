#include "deleteableleaf.h"
#include "sismagedbmanager.h"
#include "SeismicManager.h"
#include "datasetrelatedstorageimpl.h"

#include <QDir>
#include <QFileInfo>
#include <QMessageBox>
#include <QDebug>

DeletableLeaf::DeletableLeaf(const QString& name, const QStringList& paths, const QString& parentName, bool parentExists) :
		m_name(name), m_paths(paths), m_parentName(parentName), m_parentExists(parentExists) {

}

DeletableLeaf::DeletableLeaf(const DeletableLeaf& other) : m_name(other.m_name), m_paths(other.m_paths),
		m_parentName(other.m_parentName), m_parentExists(other.m_parentExists) {

}

DeletableLeaf::DeletableLeaf() : m_name(""), m_parentName(""), m_parentExists(false) {

}

DeletableLeaf::~DeletableLeaf() {

}

QString DeletableLeaf::name() const {
	return m_name;
}

QStringList DeletableLeaf::paths() const {
	return m_paths;
}

bool DeletableLeaf::isValid() const {
	std::size_t index = 0;
	while (index<m_paths.count() && QFileInfo(m_paths[index]).exists()) {
		index++;
	}
	bool result = index>=m_paths.count();
	return result;
}

bool DeletableLeaf::isCorrupted() const {
	return false;
}

QString DeletableLeaf::owner() const {
	QDateTime birthDate;
	QString owner;
	if (m_paths.count()>0) {
		QFileInfo initFilenfo(m_paths[0]);
		birthDate = initFilenfo.birthTime();
		owner = initFilenfo.owner();
		for (const QString& filepath : m_paths) {
			QFileInfo fileinfo(filepath);
			QDateTime birthDateFile = fileinfo.birthTime();
			if (birthDateFile<birthDate) {
				birthDate = birthDateFile;
				owner = fileinfo.owner();
			}
		}
	}
	return owner;
}

QDateTime DeletableLeaf::birthDate() const {
	QDateTime birthDate;
	if (m_paths.count()>0) {
		birthDate = QFileInfo(m_paths[0]).birthTime();
		for (const QString& filepath : m_paths) {
			QDateTime birthDateFile = QFileInfo(filepath).birthTime();
			if (birthDateFile<birthDate) {
				birthDate = birthDateFile;
			}
		}
	}
	return birthDate;
}

bool DeletableLeaf::isLoneChild() const {
	return !m_parentExists;
}

QString DeletableLeaf::parentName() const {
	return m_parentName;
}

// to move somewhere else like in the util folder
/*bool deleteFile(const QString& filePath) {
	QFileInfo fileInfo(filePath);
	QDir parentDir;
	bool res = fileInfo.exists() && fileInfo.isFile();
	if (!res) {
		qDebug() << "File to delete does not exists, no deletion to be done";
	} else {
		parentDir = fileInfo.dir();
		res = parentDir.exists();

		if (!res) {
			qDebug() << "Parent directory does not exists, no deletion to be done";
		} else {
			res = parentDir.remove(fileInfo.fileName());
		}
	}
	return res;
}*/

QList<DeletableLeaf> DeletableLeaf::findLeavesFromRGT2RGBDirectory(const QString& dirPath) {
	QList<DeletableLeaf> outputList;
	QDir dir(dirPath);

	QDir dirForPathExtraction(dir);
	bool tryGetParentDir = dirForPathExtraction.cdUp();
	QString sismageName = dirForPathExtraction.dirName();
	tryGetParentDir = tryGetParentDir && dirForPathExtraction.cdUp() && dirForPathExtraction.cdUp() && dirForPathExtraction.cdUp();
	if (!tryGetParentDir) {
		return outputList;
	}

	QString datasetPathFromIJK = DatasetRelatedStorageImpl::getDatasetPath(dirForPathExtraction.absolutePath(), sismageName);
	bool parentExists = !datasetPathFromIJK.isNull()&& !datasetPathFromIJK.isEmpty();

	QFileInfoList list = dir.entryInfoList(QStringList(), QDir::Files);
	for (const QFileInfo& fileInfo : list) {
		QString name = fileInfo.baseName() + " (" + fileInfo.suffix() + ")";
		QStringList paths;
		paths << fileInfo.absoluteFilePath();
		outputList.push_back(DeletableLeaf(name, paths, sismageName, parentExists));
	}
	return outputList;
}

QList<DeletableLeaf> DeletableLeaf::findLeavesFromHorizonDirectory(const QString& dirPath) {
	QList<DeletableLeaf> outputList;
	QDir dir(dirPath);

	QDir dirForPathExtraction(dir);
	bool tryGetParentDir = dirForPathExtraction.cdUp();
	QString sismageName = dirForPathExtraction.dirName();
	tryGetParentDir = tryGetParentDir&& dirForPathExtraction.cdUp() && dirForPathExtraction.cdUp() && dirForPathExtraction.cdUp();
	if (!tryGetParentDir) {
		return outputList;
	}

	QString datasetPathFromIJK = DatasetRelatedStorageImpl::getDatasetPath(dirForPathExtraction.absolutePath(), sismageName);
	bool parentExists = !datasetPathFromIJK.isNull()&& !datasetPathFromIJK.isEmpty();

	QFileInfoList list = dir.entryInfoList(QStringList() << "*.raw", QDir::Files);
	for (const QFileInfo& fileInfo : list) {
		QString name = fileInfo.baseName();
		QStringList paths = dir.entryList(QStringList() << fileInfo.fileName() + "*" << fileInfo.completeBaseName() + ".*", QDir::Files);
		for (std::size_t idx=0; idx<paths.count(); idx++) {
			paths[idx] = dir.absoluteFilePath(paths[idx]);
		}
		outputList.push_back(DeletableLeaf(name, paths, sismageName, parentExists));
	}
	return outputList;
}

QString DeletableLeaf::getDatasetDirPathFromImportExportSeismicFolder(const QString& seismicDirPath) {
	// location surveyDir/ImportExport/IJK/datasetSismageName : need 3 dir.cdUp()
	QString datasetDirPath;
	QDir dir(seismicDirPath);
	bool ok = dir.cdUp() && dir.cdUp() && dir.cdUp();
	if (ok) {
		datasetDirPath = QString::fromStdString(SismageDBManager::surveyPath2DatasetPath(dir.absolutePath().toStdString()));
	}
	return datasetDirPath;
}

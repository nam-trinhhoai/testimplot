#include "datasetrelatedstorageimpl.h"
#include "sismagedbmanager.h"
#include "SeismicManager.h"

#include <QDebug>

QString DatasetRelatedStorageImpl::CUBE_NAME = "cube.xt";

DatasetRelatedStorageImpl::DatasetRelatedStorageImpl(QString surveyPath, QString datasetSismageName) {
	m_surveyPath = surveyPath;
	m_datasetSismageName = datasetSismageName;

	m_implDir = m_surveyPath + "/ImportExport/IJK/" + m_datasetSismageName;
	m_datasetLinkPath = m_implDir + "/" + CUBE_NAME;

	checkDataset();
}

DatasetRelatedStorageImpl::~DatasetRelatedStorageImpl() {

}

QString DatasetRelatedStorageImpl::surveyPath() const {
	return m_surveyPath;
}

QString DatasetRelatedStorageImpl::datasetSismageName() const {
	return m_datasetSismageName;
}

QString DatasetRelatedStorageImpl::datasetLinkPath() const {
	return m_datasetLinkPath;
}

QString DatasetRelatedStorageImpl::implDir() const {
	return m_implDir;
}

void DatasetRelatedStorageImpl::checkDataset() {
	bool createLink = false;
	bool searchDatasetPath = true;

	QFileInfo implFileInfo(m_implDir);
	if (QDir(m_implDir).exists() && implFileInfo.isReadable() && implFileInfo.isExecutable()) {
		QFileInfo datasetLinkInfo(m_datasetLinkPath);
		if (!datasetLinkInfo.exists() && !datasetLinkInfo.isSymLink()) {
			createLink = true;
		} else if (!datasetLinkInfo.isSymbolicLink()) {
			qDebug() << "DatasetRelatedStorageImpl::checkDataset link dataset is not a link";
		} else if (!QFileInfo(datasetLinkInfo.symLinkTarget()).exists()) {
			qDebug() << "DatasetRelatedStorageImpl::checkDataset linked dataset target does not exists";
		} else if (!QFileInfo(datasetLinkInfo.symLinkTarget()).isReadable()) {
			qDebug() << "DatasetRelatedStorageImpl::checkDataset linked dataset target is not readable";
		} else if (!QFileInfo(datasetLinkInfo.symLinkTarget()).isFile()) {
			qDebug() << "DatasetRelatedStorageImpl::checkDataset linked dataset target in not a file";
		} else if (SeismicManager::filextGetAxis(datasetLinkInfo.symLinkTarget())==2) {
			qDebug() << "DatasetRelatedStorageImpl::checkDataset linked dataset target in not a valid volume";
		} else {
			searchDatasetPath = false; // link is valid
		}
	}

	if (searchDatasetPath) {
		QString datasetRealPath = getDatasetRealPath(m_surveyPath, m_datasetSismageName);

		if (!datasetRealPath.isNull() && !datasetRealPath.isEmpty()) {
			qDebug() << "DatasetRelatedStorageImpl::checkDataset failed to find the dataset";
		}

		if (createLink && !datasetRealPath.isNull() && !datasetRealPath.isEmpty()) {
			QDir implDir = QFileInfo(m_datasetLinkPath).dir();
			QString relativePath = implDir.relativeFilePath(datasetRealPath);
			QFile(relativePath).link(m_datasetLinkPath);
		} else {
			m_datasetLinkPath = datasetRealPath;
		}
	}
}

QString DatasetRelatedStorageImpl::getDatasetRealPath(QString survey, QString sismageName) {
	QString datasetPath;

	// search for sismage dataset that match
	QString datasetPathDir = QString::fromStdString(SismageDBManager::surveyPath2DatasetPath(survey.toStdString()));
	// search dataset
	// try simple solution
	QString simpleFilePath = datasetPathDir + "/seismic3d." + sismageName + ".xt";
	QFileInfo simpleFileInfo(simpleFilePath);
	bool parentExists = false;
	if (simpleFileInfo.exists() && sismageName.compare(SeismicManager::seismicFullFilenameToTinyName(simpleFilePath))==0) {
		datasetPath = simpleFileInfo.absoluteFilePath();
	} else {
		QFileInfoList listDescs = QDir(datasetPathDir).entryInfoList(QStringList()<<"*.desc", QDir::Files);
		std::size_t index = 0;
		while (!parentExists && index<listDescs.count()) {
			QString xtFile = datasetPathDir + "/" + listDescs[index].completeBaseName() + ".xt";
			if (QFileInfo(xtFile).exists() && sismageName.compare(SeismicManager::seismicFullFilenameToTinyName(xtFile))==0) {
				datasetPath = xtFile;
				parentExists = true;
			} else {
				index++;
			}
		}
	}
	return datasetPath;
}

QString DatasetRelatedStorageImpl::getDatasetPath(QString survey, QString sismageName) {
	DatasetRelatedStorageImpl impl(survey, sismageName);
	return impl.datasetLinkPath();
}


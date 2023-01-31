#ifndef SLICER_UTIL_DATASETRELATEDSTORAGEIMPL_H_
#define SLICER_UTIL_DATASETRELATEDSTORAGEIMPL_H_

#include <QString>

class DatasetRelatedStorageImpl {
public:
	DatasetRelatedStorageImpl(QString surveyPath, QString datasetSismageName);
	~DatasetRelatedStorageImpl();

	QString surveyPath() const;
	QString datasetSismageName() const;
	QString datasetLinkPath() const;
	QString implDir() const;

	// may return empty string
	// may be improved by providing hints with loaded datasets
	static QString getDatasetRealPath(QString surveyDir, QString datasetSismageName);

	// use CUBE_NAME if possible else use above function
	static QString getDatasetPath(QString surveyDir, QString datasetSismageName);

private:
	void checkDataset();

	QString m_surveyPath;
	QString m_datasetSismageName;
	QString m_implDir;
	QString m_datasetLinkPath;

	static QString CUBE_NAME;
};

#endif

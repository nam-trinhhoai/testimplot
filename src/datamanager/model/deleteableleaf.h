#ifndef DATAMANAGER_DELETEABLELEAF_H_
#define DATAMANAGER_DELETEABLELEAF_H_

#include <QString>
#include <QStringList>
#include <QDateTime>

class DeletableLeaf {
public:
	DeletableLeaf(const QString& name, const QStringList& paths, const QString& parentName, bool parentExists);
	DeletableLeaf(const DeletableLeaf& other);
	DeletableLeaf(); // for QMap
	~DeletableLeaf();

	QString name() const;
	QStringList paths() const;
	bool isValid() const;
	bool isCorrupted() const;
	QString owner() const;
	QDateTime birthDate() const;
	bool isLoneChild() const;
	QString parentName() const;

	static QList<DeletableLeaf> findLeavesFromRGT2RGBDirectory(const QString& dirPath);
	static QList<DeletableLeaf> findLeavesFromHorizonDirectory(const QString& dirPath);
	// return path if possible else an empty string can be expected
	static QString getDatasetDirPathFromImportExportSeismicFolder(const QString& seismicDirPath);

private:

	QString m_name;
	QStringList m_paths;
	QString m_parentName;
	bool m_parentExists;
};

#endif

#ifndef IFileBasedData_H
#define IFileBasedData_H

#include <QDebug>

class IFileBasedData {
public:
	IFileBasedData(QString idPath) : m_idPath(idPath) {}
	virtual ~IFileBasedData() {}

	bool isIdPathIdentical(QString idPath) {
		return m_idPath.compare(idPath)==0;
	}

	const QString& idPath() const {
		return m_idPath;
	}

private:
	QString m_idPath;
};

#endif

#ifndef SRC_BNNI_BNNICONFIG_H
#define SRC_BNNI_MODELS_BNNICONFIG_H

#include "structures.h"

#include <QObject>

class NetworkParametersModel;


class BnniConfig : public QObject {
	Q_OBJECT
public:
	BnniConfig(QObject* parent=nullptr);
	~BnniConfig();

	NetworkParametersModel* getNetworkModel();

	QString getWorkDir() const;
	void setWorkDir(const QString& path);

signals:
	void workDirChanged(QString path);

private:
	NetworkParametersModel* m_networkModel;

	QString m_workDir = "";
};

#endif

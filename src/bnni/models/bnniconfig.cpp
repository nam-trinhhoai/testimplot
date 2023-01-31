#include "bnniconfig.h"

#include "networkparametersmodel.h"

BnniConfig::BnniConfig(QObject* parent) : QObject(parent) {
	m_networkModel = new NetworkParametersModel(this);
}

BnniConfig::~BnniConfig() {

}

NetworkParametersModel* BnniConfig::getNetworkModel() {
	return m_networkModel;
}

QString BnniConfig::getWorkDir() const {
	return m_workDir;
}

void BnniConfig::setWorkDir(const QString& path) {
	if (m_workDir.compare(path)!=0) {
		m_workDir = path;
		m_networkModel->setCheckpointDir(m_workDir);
		emit workDirChanged(m_workDir);
	}
}


#include "seainterpretationserver.h"

#include <QDebug>

const char* SeaInterpretationServer::PROGRAM = "SEAInterpretationAPI-22.3.0-SNAPSHOT-runner.jar";

SeaInterpretationServer::SeaInterpretationServer(int port, const QString& translationServerAddress,
		const QString& dirProject)  : SeaServer(port) {
	m_translationServerAddress = translationServerAddress;
	m_dirProject = dirProject;
	m_isJavaLauncher = true;
}

SeaInterpretationServer::~SeaInterpretationServer() {

}

QString SeaInterpretationServer::program() const {
	return QString(PROGRAM);
}

QStringList SeaInterpretationServer::options() const {
	QStringList out;
	out << ("-Dsismage.dir-project=" + m_dirProject);
	out << ("-Dquarkus.http.port=" + QString::number(port()));
	out << ("-Did-translator-api/mp-rest/url=" + m_translationServerAddress);
	return out;
}

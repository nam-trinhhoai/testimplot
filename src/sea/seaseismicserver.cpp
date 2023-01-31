#include "seaseismicserver.h"

#include <QDebug>

const char* SeaSeismicServer::PROGRAM = "SEASeismicAPI-22.1.0-SNAPSHOT-runner";

SeaSeismicServer::SeaSeismicServer(int port, const QString& translationServerAddress,
		const QString& dirProject)  : SeaServer(port) {
	m_translationServerAddress = translationServerAddress;
	m_dirProject = dirProject;
	m_isJavaLauncher = false;
}

SeaSeismicServer::~SeaSeismicServer() {

}

QString SeaSeismicServer::program() const {
	return QString(PROGRAM);
}

QStringList SeaSeismicServer::options() const {
	QStringList out;
	out << ("-Dsismage.dir-project=" + m_dirProject);
	out << ("-Dquarkus.http.port=" + QString::number(port()));
	out << ("-Did-translator-api/mp-rest/url=" + m_translationServerAddress);
	return out;
}

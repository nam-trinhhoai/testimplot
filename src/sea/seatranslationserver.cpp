#include "seatranslationserver.h"

const char* SeaTranslationServer::PROGRAM = "SEAIDTranslatorAPI-22.3.0-SNAPSHOT-runner.jar";

SeaTranslationServer::SeaTranslationServer(int port, const QString& salt) :
		SeaServer(port) {
	m_salt = salt;
	m_isJavaLauncher = true;
}

SeaTranslationServer::~SeaTranslationServer() {

}

QString SeaTranslationServer::program() const {
	return QString(PROGRAM);
}

QStringList SeaTranslationServer::options() const {
	QStringList out;
	out << "-Did-translator-config.salt=" + m_salt;
	out << "-Did-translator-config.is-dictionary-database-enabled=false";
	out << "-Dquarkus.http.port=" + QString::number(port());
	return out;
}

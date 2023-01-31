#include "seamodules.h"

#include <QDebug>

const int SeaModules::NUMBER_OF_MODULES = 3;

SeaModules::SeaModules(const std::vector<int>& ports, const QString& dirProject) {
	m_ports = ports;
	m_dirProject = dirProject;

	// hard coded salt
	QString salt = "N0wItsT4sty231456789";
	m_translationServer.reset(new SeaTranslationServer(ports[0], salt));
	m_seismicServer.reset(new SeaSeismicServer(ports[1], m_translationServer->addressWithPort(), m_dirProject));
	m_interpretationServer.reset(new SeaInterpretationServer(ports[2], m_translationServer->addressWithPort(), m_dirProject));
}

SeaModules::~SeaModules() {

}

SeaModules* SeaModules::create(const std::vector<int>& ports, const QString& dirProject) {
	SeaModules* out = nullptr;

	// check if ports are valid
	if (ports.size()==NUMBER_OF_MODULES) {
		out = new SeaModules(ports, dirProject);
	}

	return out;
}

bool SeaModules::start() {
	stopOrphanServers();

	bool valid = true;
	if (!m_translationServer->isRunning()) {
		valid = m_translationServer->start();
		if (!valid) {
			qDebug() << "SeaModules::start failed to start not running translationServer";
		}
	}
	if (valid && !m_seismicServer->isRunning()) {
		valid = m_seismicServer->start();
		if (!valid) {
			qDebug() << "SeaModules::start failed to start not running seismicServer";
			stop();
		}
	}
	if (valid && !m_interpretationServer->isRunning()) {
		valid = m_interpretationServer->start();
		if (!valid) {
			qDebug() << "SeaModules::start failed to start not running interpretationServer";
			stop();
		}
	}
	return valid;
}

bool SeaModules::stop() {
	bool valid = true;
	if (m_interpretationServer->isRunning()) {
		valid = m_interpretationServer->stop();
		if (!valid) {
			qDebug() << "SeaModules::stop failed to stop running interpretationServer";
		}
	}
	if (m_seismicServer->isRunning()) {
		valid = m_seismicServer->stop();
		if (!valid) {
			qDebug() << "SeaModules::stop failed to stop running seismicServer";
		}
	}
	if (valid && m_translationServer->isRunning()) {
		valid = m_translationServer->stop();
		if (!valid) {
			qDebug() << "SeaModules::stop failed to stop running translationServer";
		}
	}
	return valid;
}


QString SeaModules::translationAddressWithPort() const {
	return m_translationServer->addressWithPort();
}

QString SeaModules::seismicAddressWithPort() const {
	return m_seismicServer->addressWithPort();
}

QString SeaModules::seismicAddress() const {
	return m_seismicServer->address();
}

int SeaModules::seismicPort() const {
	return m_seismicServer->port();
}

QString SeaModules::seismicScheme() const {
	return m_seismicServer->scheme();
}

QString SeaModules::interpretationAddressWithPort() const {
	return m_interpretationServer->addressWithPort();
}

QString SeaModules::interpretationAddress() const {
	return m_interpretationServer->address();
}

int SeaModules::interpretationPort() const {
	return m_interpretationServer->port();
}

QString SeaModules::interpretationScheme() const {
	return m_interpretationServer->scheme();
}

void SeaModules::stopOrphanServers() {
	std::vector<const char*> programs = {SeaTranslationServer::PROGRAM,
			SeaSeismicServer::PROGRAM, SeaInterpretationServer::PROGRAM};

	QProcess process;
	process.start("ps", QStringList() << "-f");
	process.waitForFinished();
	QString standardStr(process.readAllStandardOutput());
	QStringList splitStandardOutput = standardStr.split("\n", Qt::SkipEmptyParts);

	// first line is the header
	QStringList pids;
	for (int outputIdx = 1; outputIdx<splitStandardOutput.size(); outputIdx++) {
		// ps output separator seem to be one or more spaces
		QStringList splittedLine = splitStandardOutput[outputIdx].split(" ", Qt::SkipEmptyParts);
		// output format is :
		// UID PID PPID C STIME TTY TIME CMD
		if (splittedLine.size()<8) {
			continue;
		}

		bool ok;
		int ppid = splittedLine[2].toInt(&ok);
		if (!ok) {
			continue;
		}
		int pid = splittedLine[1].toInt(&ok);
		if (!ok) {
			continue;
		}

		QStringList cmdLineArgs(splittedLine.begin()+7, splittedLine.end());
		QString cmdLine = cmdLineArgs.join(" ");

		bool found = false;
		int programIdx = 0;
		while (!found && programIdx<programs.size()) {
			found = found && ppid==1 && cmdLine.contains(programs[programIdx]);
			programIdx++;
		}
		if (found) {
			pids.append(QString::number(pid));
		}
	}

	if (pids.size()>0) {
		process.start("kill", pids);
		process.waitForFinished();
	}
}

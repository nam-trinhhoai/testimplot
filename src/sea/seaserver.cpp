#include "seaserver.h"

#include <chrono>

#include <QDebug>
#include <QDir>
#include <QGuiApplication>

SeaServer::SeaServer(int port) {
	m_port = port;
	m_process.setReadChannel(QProcess::StandardOutput);

	m_serverSaidStarted = false;
}

SeaServer::~SeaServer() {

}

bool SeaServer::start() {
	bool valid = false;
	if (m_process.state()==QProcess::NotRunning) {
		// flush old data to allow correct detection of server start
		if (m_process.isOpen()) {
			m_process.readAll();
		}
		m_serverSaidStarted = false;

		if (m_isJavaLauncher) {
			QDir binDir = QFileInfo(QGuiApplication::applicationFilePath()).dir();

			QStringList launchOptions = options();
			launchOptions << "-jar" << binDir.absoluteFilePath(program());
			if (m_debugLevel>0) {
				qDebug() << "SeaServer::startServer start process " << "java" << launchOptions;
			}
			m_process.start("java", launchOptions);
		} else {
			if (m_debugLevel>0) {
				qDebug() << "SeaServer::startServer start process " << program() << options();
			}
			m_process.start(program(), options());
		}
		// wait for the command to be running
		m_process.waitForStarted(-1);

		valid = isRunning();
		if (m_debugLevel>0) {
			qDebug() << "SeaServer::startServer server post wait for started : " << m_process.state();
		}


		if (valid) {
			// wait for server to say it is ready
			isServerReady();
			valid = isRunning();
			if (m_debugLevel>0) {
				qDebug() << "SeaServer::startServer server running : " << m_process.state();
			}
		}
	}

	return valid;
}

bool SeaServer::stop() {
	bool valid = false;
	if (m_process.state()==QProcess::Running) {
		m_process.kill();
		m_process.waitForFinished(-1);

		valid = m_process.state()==QProcess::NotRunning;
	}
	return valid;
}

QString SeaServer::addressWithPort() const {
	return scheme() + "://" + address() + ":" + QString::number(port());
}

QString SeaServer::address() const {
	return "localhost";
}

int SeaServer::port() const {
	return m_port;
}

QString SeaServer::scheme() const {
	return "http";
}

bool SeaServer::isRunning() const {
	return m_process.state()==QProcess::Running;
}

bool SeaServer::isServerReady() {
	if (!m_serverSaidStarted) {
		std::chrono::steady_clock::time_point startPt = std::chrono::steady_clock::now();
		long oriTime = 4000; // wait for 4s
		long time = oriTime;
		bool gotNoData = !m_process.isReadable(); // try to read available data first
		while (time>0 && !m_serverSaidStarted && isRunning()) {
			bool gotAnswer = true;
			if (gotNoData) {
				gotAnswer = m_process.waitForReadyRead(time);
			} else {
				gotNoData = true;
			}
			if (gotAnswer) { // only try if there is data to test
				QByteArray array = m_process.readAll();
				QString str = QString::fromUtf8(array); // use default encoding

				// try to detect "started" keyword
				m_serverSaidStarted = str.contains("started", Qt::CaseInsensitive);
			}
			// update time to avoid waiting too long
			std::chrono::steady_clock::time_point endPt = std::chrono::steady_clock::now();
			time = oriTime - std::chrono::duration<double, std::milli>(endPt - startPt).count();
		}
	}
	return m_serverSaidStarted;
}

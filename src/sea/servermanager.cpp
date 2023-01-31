#include "servermanager.h"
#include "seamodules.h"

#include <QDebug>
#include <QFile>
#include <QMutexLocker>
#include <QProcess>
#include <QTextStream>

#include <QRegularExpression>


std::unique_ptr<ServerManager> ServerManager::serverManager;

// see https://unix.stackexchange.com/questions/168794/how-to-get-the-list-of-ports-which-are-free-in-a-unix-server/168804
int ServerManager::nextPort = 32767;
const int ServerManager::startPort = 31000; // arbitrary
const int ServerManager::endPort = 32767;
long ServerManager::nextKey = 1;
const long ServerManager::INVALID_SERVERS_KEY = 0;
bool ServerManager::SERVICES_READ = false;
std::list<int> ServerManager::SERVICES_PORTS = std::list<int>();


ServerManager::ServerManager() {

}

ServerManager::~ServerManager() {

}

long ServerManager::startServer(const QString& dirProject) {
	QMutexLocker locker(&m_mutex);
	std::map<QString, ServersHolder>::const_iterator it = m_dirProjectToServer.find(dirProject);
	bool createServers = it==m_dirProjectToServer.end();

	long key = INVALID_SERVERS_KEY;
	int numberOfModules = SeaModules::NUMBER_OF_MODULES;
	std::vector<int> ports;
	if (createServers) {
		ports = getPortsForServer(numberOfModules);
		if (m_debugLevel>0) {
			qDebug() << "ServerManager::startServer got ports: " << ports << ", got " << ports.size() << ", expected " << numberOfModules;
		}
		createServers = ports.size()==numberOfModules;
	} else {
		key = nextKey++;
		m_dirProjectToServer[dirProject].accessKeys.push_back(key);
	}
	if (createServers) {
		std::unique_ptr<SeaModules> server;
		server.reset(SeaModules::create(ports, dirProject));
		bool serversCreated = server->start();
		if (m_debugLevel>0) {
			qDebug() << "ServerManager::startServer server->start() returned " << serversCreated;
		}
		if (serversCreated) {
			m_dirProjectToServer[dirProject].servers = std::move(server);
			key = nextKey++;
			m_dirProjectToServer[dirProject].accessKeys.push_back(key);
		} else {
			qDebug() << "ServerManager::startServer failed to start server";
		}
	}
	return key;
}

bool ServerManager::stopServer(const QString& dirProject, long serversKey) {
	QMutexLocker locker(&m_mutex);
	std::map<QString, ServersHolder>::iterator it = m_dirProjectToServer.find(dirProject);
	bool out = it!= m_dirProjectToServer.end();
	if (out) {
		std::list<long>::iterator keyIt = std::find(it->second.accessKeys.begin(), it->second.accessKeys.end(), serversKey);
		out = keyIt!=it->second.accessKeys.end();
		if (out) {
			it->second.accessKeys.erase(keyIt);

			// only stop servers if there is no key left
			if (it->second.accessKeys.size()==0) {
				out = it->second.servers->stop();
				if (!out) {
					qDebug() << "ServerManager::stopServer failed to stop server";
				}
				m_dirProjectToServer.erase(it);
			}
		}
	}
	return out;
}

bool ServerManager::contains(const QString& dirProject) {
	QMutexLocker locker(&m_mutex);
	std::map<QString, ServersHolder>::iterator it = m_dirProjectToServer.find(dirProject);
	return it!=m_dirProjectToServer.end();
}

SeaModules* ServerManager::server(const QString& dirProject) {
	QMutexLocker locker(&m_mutex);
	std::map<QString, ServersHolder>::iterator it = m_dirProjectToServer.find(dirProject);
	SeaModules* out = nullptr;
	if (it!=m_dirProjectToServer.end()) {
		out = it->second.servers.get();
	}
	return out;
}

ServerManager& ServerManager::getServerManager() {
	if (serverManager==nullptr) {
		serverManager.reset(new ServerManager());
	}
	return *serverManager;
}

bool ServerManager::isPortFree(int port, std::list<int> blockedPorts) {
	if (!SERVICES_READ) {
		readServices();
	}

	// avoid port of known services
	auto it = std::find(SERVICES_PORTS.begin(), SERVICES_PORTS.end(), port);
	if (it!=SERVICES_PORTS.end()) {
		return false;
	}

	auto itBlocked = std::find(blockedPorts.begin(), blockedPorts.end(), port);
	bool valid = itBlocked==blockedPorts.end();

	return valid;
}

// May be good to use : https://unix.stackexchange.com/questions/168794/how-to-get-the-list-of-ports-which-are-free-in-a-unix-server/168804
std::vector<int> ServerManager::getPortsForServer(int numberOfRequestedPorts) {
	std::vector<int> ports;
	ports.resize(numberOfRequestedPorts);

	std::list<int> blockedPorts = netstatBlockedPorts();

	int loopStartPort = nextPort;
	bool notLooped = true;
	int i=0;
	while (notLooped && i<numberOfRequestedPorts) {
		bool portNotFound = true;
		while (notLooped && portNotFound) {
			int port = nextPort;
			nextPort -= 1;
			if (nextPort<startPort) {
				nextPort = endPort;
			}

			portNotFound = !isPortFree(port, blockedPorts);
			ports[i] = port;

			notLooped = loopStartPort!=nextPort;
		}

		i++;
	}
	if (!notLooped && i<numberOfRequestedPorts) {
		qDebug() << "ServerManager::getPortsForServer : Not enough available ports for servers";
	}

	return ports;
}

std::list<int> ServerManager::netstatBlockedPorts() {
	std::list<int> ports;

	QProcess process;
	process.start("netstat", QStringList()<<"-an");
	process.waitForFinished(-1);

	QTextStream stream(&process);
	QString line;
	while (stream.readLineInto(&line)) {
		if (!line.contains("LISTEN")) {
			continue;
		}

		QStringList lineSplit = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);

		if (lineSplit.size()<4) {
			continue;
		}
		QString ipAndPort = lineSplit[3];
		QStringList portList = ipAndPort.split(":");
		if (portList.size()<2) {
			continue;
		}
		QString portStr = portList[1];
		bool intConversion;
		int port = portStr.toInt(&intConversion);
		if (intConversion) {
			ports.push_back(port);
		}
	}
	return ports;
}

void ServerManager::readServices() {
	QFile serviceFile("/etc/services");

	SERVICES_PORTS.clear();
	if (serviceFile.open(QFile::ReadOnly | QFile::Text | QFile::ExistingOnly)) {
		QTextStream stream(&serviceFile);
		QString line;
		while (stream.readLineInto(&line)) {
			bool lineEmpty = line.isNull() || line.isEmpty();
			if (lineEmpty || line[0]=='#') {
				continue;
			}

			QStringList lineSplit = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);

			if (lineSplit.size()<2) {
				continue;
			}
			QString portWithProtocol = lineSplit[1];
			QStringList portSplit = portWithProtocol.split("/");
			if (portSplit.size()==0) {
				continue;
			}
			QString portStr = portSplit[0];
			bool intConversion;
			int port = portStr.toInt(&intConversion);
			if (intConversion) {
				SERVICES_PORTS.push_back(port);
			}
		}
	}
	SERVICES_READ = true;
}

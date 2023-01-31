#ifndef SRC_SEA_SERVERMANAGER_H
#define SRC_SEA_SERVERMANAGER_H

#include "seamodules.h"

#include <QMutex>
#include <QString>

#include <map>
#include <vector>
#include <list>
#include <memory>

class ServerManager {
public:
	struct ServersHolder {
		std::unique_ptr<SeaModules> servers;
		std::list<long> accessKeys;
	};

	ServerManager();
	~ServerManager();

	long startServer(const QString& dirProject);
	bool stopServer(const QString& dirProject, long serversKey);

	bool contains(const QString& dirProject);
	SeaModules* server(const QString& dirProject);

	static ServerManager& getServerManager();


	static std::vector<int> getPortsForServer(int numberOfRequestedPorts);
	static bool isPortFree(int port, std::list<int> blockedPorts);
	static std::list<int> netstatBlockedPorts();
	static void readServices();

	static const long INVALID_SERVERS_KEY;

private:
	static std::unique_ptr<ServerManager> serverManager;
	static int nextPort;
	static const int startPort;
	static const int endPort;
	static long nextKey;

	static bool SERVICES_READ;
	static std::list<int> SERVICES_PORTS;
	std::map<QString, ServersHolder> m_dirProjectToServer;

	QMutex m_mutex;

	int m_debugLevel = 0;
};

#endif

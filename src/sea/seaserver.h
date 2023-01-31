#ifndef SRC_SEA_SEASERVER_H
#define SRC_SEA_SEASERVER_H

#include <QProcess>

class SeaServer {
public:
	SeaServer(int port);
	virtual ~SeaServer();

	bool start();
	bool stop();

	QString addressWithPort() const;
	QString address() const;
	int port() const;
	QString scheme() const;
	bool isRunning() const;

protected:
	virtual QString program() const = 0;
	virtual QStringList options() const = 0;
	virtual bool isServerReady();

	bool m_serverSaidStarted;
	bool m_isJavaLauncher;

private:
	QProcess m_process;
	int m_port;

	int m_debugLevel = 0;
};

#endif

#ifndef SRC_SEA_SEASEISMICSERVER_H
#define SRC_SEA_SEASEISMICSERVER_H

#include "seaserver.h"

class SeaSeismicServer : public SeaServer {
public:
	SeaSeismicServer(int port, const QString& translationServerAddress, const QString& dirProject);
	virtual ~SeaSeismicServer();

	static const char* PROGRAM;

protected:
	virtual QString program() const override;
	virtual QStringList options() const override;

private:
	QString m_translationServerAddress;
	QString m_dirProject;
};

#endif

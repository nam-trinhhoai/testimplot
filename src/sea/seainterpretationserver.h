#ifndef SRC_SEA_SEAINTERPRETATIONSERVER_H
#define SRC_SEA_SEAINTERPRETATIONSERVER_H

#include "seaserver.h"

class SeaInterpretationServer : public SeaServer {
public:
	SeaInterpretationServer(int port, const QString& translationServerAddress, const QString& dirProject);
	virtual ~SeaInterpretationServer();

	static const char* PROGRAM;

protected:
	virtual QString program() const override;
	virtual QStringList options() const override;

private:
	QString m_translationServerAddress;
	QString m_dirProject;
};

#endif

#ifndef SRC_SEA_SEATRANSLATIONSERVER_H
#define SRC_SEA_SEATRANSLATIONSERVER_H

#include "seaserver.h"

class SeaTranslationServer : public SeaServer {
public:
	SeaTranslationServer(int port, const QString& salt);
	virtual ~SeaTranslationServer();

	static const char* PROGRAM;

protected:
	virtual QString program() const override;
	virtual QStringList options() const override;

private:
	QString m_salt;
};

#endif

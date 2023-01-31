#ifndef SRC_SEA_SEAMODULES_H
#define SRC_SEA_SEAMODULES_H

#include "seatranslationserver.h"
#include "seaseismicserver.h"
#include "seainterpretationserver.h"

#include <QString>

#include <vector>
#include <memory>

class SeaModules {
public:
	static const int NUMBER_OF_MODULES;

	~SeaModules();

	bool start();
	bool stop();

	QString translationAddressWithPort() const;
	QString seismicAddressWithPort() const;
	QString seismicAddress() const;
	int seismicPort() const;
	QString seismicScheme() const;
	QString interpretationAddressWithPort() const;
	QString interpretationAddress() const;
	int interpretationPort() const;
	QString interpretationScheme() const;

	static SeaModules* create(const std::vector<int>& ports, const QString& dirProject);

	static void stopOrphanServers();
private:
	SeaModules(const std::vector<int>& ports, const QString& dirProject);

	QString m_dirProject;
	std::vector<int> m_ports;
	std::unique_ptr<SeaTranslationServer> m_translationServer;
	std::unique_ptr<SeaSeismicServer> m_seismicServer;
	std::unique_ptr<SeaInterpretationServer> m_interpretationServer;
};

#endif

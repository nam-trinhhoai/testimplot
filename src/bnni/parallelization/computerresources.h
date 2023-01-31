#ifndef SRC_BNNI_PARALLELIZATION_COMPUTERRESOURCES_H
#define SRC_BNNI_PARALLELIZATION_COMPUTERRESOURCES_H

#include <QObject>
#include <QProcess>

#include <vector>
#include <map>

#include "sshfingerprint.h"

class ComputerResource : public QObject {
	Q_OBJECT
public:
	typedef struct GPUInfo {
		long totalMemory = 0;
		long freeMemory = 0;
	} GPUInfo;

	ComputerResource(const QString& hostname, QObject* parent=0);
	~ComputerResource();

	// computer properties
	QString hostName() const;
	bool isAvailable() const;
	unsigned int numberOfGPUs() const;
	GPUInfo gpuInfo(unsigned int gpuNumber) const;

	// return finger print from output of "ssh-keygen -l -f
	std::vector<SshFingerprintAndKey> getSshFingerprint() const;

	static ComputerResource* getCurrentComputer(QObject* parent=0);

signals:
	void statusChanged();
	void gpuStatusChanged();

public slots:
	void updateStatus();
	void updateGpuInfos();

private:
	void resetGPUs();

	QString m_hostname;
	bool m_isAvailable;
	unsigned int m_numberOfGPUs = 0;
	std::vector<GPUInfo> m_gpuInfos;
};

class ComputerResources : public QObject {
	Q_OBJECT
public:
	ComputerResources(QObject* parent=0);
	~ComputerResources();

	// ComputerResources take ownership of created objects
	bool addResourcesFromFile(const QString& path);

	// ComputerResources take ownership of added objects
	bool addResource(ComputerResource* resource);

	// caller should take ownership of the removed object
	bool removeResource(ComputerResource* resource);
	bool removeAndDeleteResource(ComputerResource* resource);

	const std::vector<ComputerResource*>& resources() const;

signals:
	void resourceAdded(ComputerResource* resource);
	void resourceRemoved(ComputerResource* resource);
private:
	std::vector<ComputerResource*> m_resources;
	std::map<ComputerResource*, QMetaObject::Connection> m_resourceConns;
};

#endif

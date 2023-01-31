#include "computerresources.h"
#include "sshhostkey.h"

#include <QProcess>
#include <QTemporaryFile>
#include <QDebug>

#include <algorithm>

ComputerResource::ComputerResource(const QString& hostname, QObject* parent) :
		QObject(parent), m_hostname(hostname) {
	m_isAvailable = false;
	updateStatus();
}

ComputerResource::~ComputerResource() {

}

void ComputerResource::updateStatus() {
	bool newIsAvailable = true;
	if (newIsAvailable!=m_isAvailable) {
		m_isAvailable = newIsAvailable;
		updateGpuInfos();
		emit statusChanged();
	}
}

void ComputerResource::updateGpuInfos() {
	if (m_isAvailable) {
		emit gpuStatusChanged();
	} else {
		resetGPUs();
	}
}

void ComputerResource::resetGPUs() {
	if (m_numberOfGPUs!=0) {
		m_numberOfGPUs = 0;
		m_gpuInfos.clear();

		emit gpuStatusChanged();
	}
}

QString ComputerResource::hostName() const {
	return m_hostname;
}

bool ComputerResource::isAvailable() const {
	return m_isAvailable;
}

unsigned int ComputerResource::numberOfGPUs() const {
	return m_numberOfGPUs;
}

ComputerResource::GPUInfo ComputerResource::gpuInfo(unsigned int gpuNumber) const {
	GPUInfo gpuInfo;
	if (gpuNumber<m_numberOfGPUs) {
		gpuInfo = m_gpuInfos[gpuNumber];
	}
	return gpuInfo;
}

ComputerResource* ComputerResource::getCurrentComputer(QObject* parent) {
	QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
	QString hostNameVar = "HOSTNAME";
	QString hostName = env.value(hostNameVar, "localhost");
	ComputerResource* resource = new ComputerResource(hostName, parent);
	return resource;
}

std::vector<SshFingerprintAndKey> ComputerResource::getSshFingerprint() const {
	std::vector<SshFingerprintAndKey> output;

	QTemporaryFile file;
	file.setAutoRemove(true);
	bool fileValid = file.open();
	file.close();

	if (fileValid) {
		QProcess process;
		process.setStandardOutputFile(file.fileName());
		process.setStandardErrorFile("/dev/null");
		process.start("ssh-keyscan", QStringList() << m_hostname);
		bool processFinished = process.waitForFinished(-1);

		process.setStandardOutputFile("");
		process.setStandardErrorFile("");
		process.setReadChannel(QProcess::StandardOutput);

		if (processFinished) {
			process.start("ssh-keygen", QStringList() << "-l" << "-f" << file.fileName());
			processFinished = process.waitForFinished(-1);
		}
		if (processFinished) {
			QByteArray standardOuput = process.readAllStandardOutput();
			QString standardOuputStr(standardOuput);
			QStringList standardOuputStrList = standardOuputStr.split("\n");

			// read file to get host key
			file.open();
			QTextStream textStream(&file);
			for (int i=0; i<standardOuputStrList.count(); i++) {
				SshFingerprintAndKey duo;
				duo.fingerprint = SshFingerprint::getFingerprintFromKeyGen(standardOuputStrList[i]);

				QString hostKey;
				bool readValid = textStream.readLineInto(&hostKey);
				if (readValid) {
					duo.hostKey = SshHostKey::getHostKey(hostKey);
				}

				if (duo.fingerprint!=nullptr && duo.hostKey!=nullptr) {
					output.push_back(duo);
				}
			}
			file.close();
		}
	}

	return output;
}

ComputerResources::ComputerResources(QObject* parent) :
		QObject(parent) {
}

ComputerResources::~ComputerResources() {
	// no need to destroy each item of m_resources because resource->parent() == this
	// clear conns to avoid being called by destroying resources during destruction
	for (const std::pair<ComputerResource*, QMetaObject::Connection>& pair : m_resourceConns) {
		disconnect(pair.second);
	}
	m_resourceConns.clear();
}

bool ComputerResources::addResourcesFromFile(const QString& path) {
	bool valid = false;
	QStringList hostNames;

	if (valid) {
		for (const QString& hostName : hostNames) {
			ComputerResource* resource = new ComputerResource(hostName, this);
			bool out = addResource(resource);
			if (!out) {
				resource->deleteLater();
			}
		}
	}
	return valid;
}

bool ComputerResources::addResource(ComputerResource* resource) {
	std::vector<ComputerResource*>::const_iterator it = std::find_if(m_resources.begin(), m_resources.end(),
			[resource](ComputerResource* other) {
		bool out = resource->hostName().compare(other->hostName())==0;
		return out;
	});
	bool valid = it==m_resources.end();
	if (valid) {
		resource->setParent(this);
		QMetaObject::Connection conn = connect(resource, &ComputerResource::destroyed, [this, resource]() {
			bool out = removeResource(resource);
			if (!out) {
				qDebug() << "ComputerResources : Failed to removed destroyed object";
			}
		});
		m_resourceConns[resource] = conn;
		m_resources.push_back(resource);
		emit resourceAdded(resource);
	}
	return valid;
}

bool ComputerResources::removeResource(ComputerResource* resource) {
	std::vector<ComputerResource*>::const_iterator it = std::find(m_resources.begin(), m_resources.end(), resource);
	bool valid = it!=m_resources.end();
	if (valid) {
		m_resources.erase(it);
		disconnect(m_resourceConns[resource]);
		m_resourceConns.erase(resource);
		resource->setParent(nullptr);
		emit resourceRemoved(resource);
	}
	return valid;
}

bool ComputerResources::removeAndDeleteResource(ComputerResource* resource) {
	bool valid = removeResource(resource);
	if (valid) {
		resource->deleteLater();
	}
	return valid;
}

const std::vector<ComputerResource*>& ComputerResources::resources() const {
	return m_resources;
}

#include "sshhostkey.h"

#include <QStringList>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QDebug>

// key from known host or output of ssh-keyscan myhost
SshHostKey::SshHostKey(const QString& hostKey, const QString& keyAlgorithm, const QString& key) {
	m_completeHostKey = hostKey;
	m_keyAlgorithm = keyAlgorithm;
	m_key = key;
}

SshHostKey::SshHostKey(const SshHostKey& other) {
	m_completeHostKey = other.m_completeHostKey;
	m_keyAlgorithm = other.m_keyAlgorithm;
	m_key = other.m_key;
}

const QString& SshHostKey::completeHostKey() const {
	return m_completeHostKey;
}

const QString& SshHostKey::keyAlgorithm() const {
	return m_keyAlgorithm;
}

const QString& SshHostKey::key() const {
	return m_key;
}

const SshHostKey& SshHostKey::operator=(const SshHostKey& other) {
	m_completeHostKey = other.m_completeHostKey;
	m_keyAlgorithm = other.m_keyAlgorithm;
	m_key = other.m_key;
	return *this;
}

std::shared_ptr<SshHostKey> SshHostKey::getHostKey(const QString& hostKey) {
	QString trimmedHostKey = hostKey.trimmed(); // to avoid \n
	QStringList list = trimmedHostKey.split(" ");
	bool valid = list.size()==3;

	std::shared_ptr<SshHostKey> out = nullptr;
	if (valid) {
		QString keyAlgo, key;
		// index 0 : ignore host part for now
		// index 1 : key generation algorithm, do not check for now
		keyAlgo = list[1];

		// index 2 : key part of hostKey
		key = list[2];

		out = std::make_shared<SshHostKey>(trimmedHostKey, keyAlgo, key);
	}
	return out;
}

bool operator==(const SshHostKey& a, const SshHostKey& b) {
	// do not test m_completeHostKey because host can vary (with or without ip or hostname or hashed)
	return a.m_keyAlgorithm.compare(b.m_keyAlgorithm)==0 &&
			a.m_completeHostKey.compare(b.m_completeHostKey)==0;
}

bool SshHostKey::addToKnownHosts() const {
	bool valid = true;
	QDir homeDir = QDir::home();
	if (!homeDir.exists()) {
		valid = false;
		qDebug() << "User home directory does not exist, cannot find or create known hosts";
	}

	QString sshDirName = ".ssh";
	if (valid && !homeDir.exists(sshDirName)) {
		valid = homeDir.mkdir(sshDirName);
		valid = valid && QFile::setPermissions(homeDir.absoluteFilePath(sshDirName),
				QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner);
	}

	if (valid) {
		QString knownHostsName = "known_hosts";
		QDir sshDir(homeDir.absoluteFilePath(sshDirName));
		QFile knownHosts(sshDir.absoluteFilePath(knownHostsName));
		bool isNewFile = !knownHosts.exists();
		valid = knownHosts.open(QIODevice::Append | QIODevice::Text);
		if (valid) {
			QTextStream textStream(&knownHosts);
			textStream << m_completeHostKey << Qt::endl;
		}
		knownHosts.close();
		if (isNewFile) {
			QFile::setPermissions(knownHosts.fileName(),
					QFileDevice::ReadOwner | QFileDevice::WriteOwner |
					QFileDevice::ReadGroup | QFileDevice::ReadOther);
		}
	}
	return valid;
}

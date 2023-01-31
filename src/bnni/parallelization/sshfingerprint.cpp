#include "sshfingerprint.h"
#include "sshhostkey.h"

#include <QTemporaryFile>
#include <QTextStream>
#include <QProcess>
#include <QRegularExpression>

SshFingerprint::SshFingerprint(const QString& hostname, const QString& ip, const QString& hash,
		KEY_ALGORITHM keyAlgorithm, int algoBitsSize) {
	m_hostname = hostname;
	m_ip = ip;
	m_keyAlgorithm = keyAlgorithm;
	m_algoBitsSize = algoBitsSize;
	m_hash = hash;
}

SshFingerprint::SshFingerprint(const SshFingerprint& fingerprint) {
	this->m_hostname = fingerprint.m_hostname;
	this->m_ip = fingerprint.m_ip;
	this->m_keyAlgorithm = fingerprint.m_keyAlgorithm;
	this->m_algoBitsSize = fingerprint.m_algoBitsSize;
	this->m_hash = fingerprint.m_hash;
}

SshFingerprint::~SshFingerprint() {

}

const SshFingerprint& SshFingerprint::operator=(const SshFingerprint& fingerprint) {
	this->m_hostname = fingerprint.m_hostname;
	this->m_ip = fingerprint.m_ip;
	this->m_keyAlgorithm = fingerprint.m_keyAlgorithm;
	this->m_algoBitsSize = fingerprint.m_algoBitsSize;
	this->m_hash = fingerprint.m_hash;

	return *this;
}

bool operator==(const SshFingerprint& a, const SshFingerprint& b) {
	bool isSame = a.m_algoBitsSize==b.m_algoBitsSize && a.m_keyAlgorithm==b.m_keyAlgorithm;
	bool hostNotDefined = true;
	if (isSame && !a.m_hostname.isNull() && !a.m_hostname.isEmpty() && !b.m_hostname.isNull() &&
			!b.m_hostname.isEmpty()) {
		hostNotDefined = false;
		isSame = isSame && a.m_hostname.compare(b.m_hostname)==0;
	}
	bool ipNotDefined = true;
	if (isSame && !a.m_ip.isNull() && !a.m_ip.isEmpty() && !b.m_ip.isNull() && !b.m_ip.isEmpty()) {
		ipNotDefined = false;
		isSame = isSame && a.m_ip.compare(b.m_ip)==0;
	}
	if (ipNotDefined && hostNotDefined) {
		isSame = false;
	}
	isSame = isSame && a.m_hash.compare(b.m_hash)==0;

	return isSame;
}

SshFingerprint::operator QString() const {
	QString hostStr;
	if (!m_hostname.isNull() && !m_hostname.isEmpty() && !m_ip.isNull() && !m_ip.isEmpty()) {
		hostStr = m_hostname + "," + m_ip;
	} else if (!m_hostname.isNull() && !m_hostname.isEmpty()) {
		hostStr = m_hostname;
	} else if (!m_ip.isNull() && !m_ip.isEmpty()) {
		hostStr = m_ip;
	}

	return QString::number(m_algoBitsSize) + " " + m_hash + " " +
			hostStr + " (" + getStringFromKeyAlgo(m_keyAlgorithm) + ")";
}

const QString& SshFingerprint::hostname() const {
	return m_hostname;
}

const QString& SshFingerprint::ip() const {
	return m_ip;
}

const QString& SshFingerprint::hash() const {
	return m_hash;
}

SshFingerprint::KEY_ALGORITHM SshFingerprint::keyAlgorithm() const {
	return m_keyAlgorithm;
}

int SshFingerprint::algoBitsSize() const {
	return m_algoBitsSize;
}

std::shared_ptr<SshFingerprint> SshFingerprint::getFingerprintFromKeyGen(const QString& fingerprint) {
	std::shared_ptr<SshFingerprint> out = nullptr;

	QStringList list = fingerprint.split(" ");
	bool valid = list.size()==4;

	int algoBitsSize;
	QString hash, hostname, ip;
	if (valid) { // extract algoBitsSize and hash
		algoBitsSize = list[0].toInt(&valid);

		hash = list[1];
		valid = valid && !hash.isNull() && !hash.isEmpty();
	}

	if (valid) { // extract host and ip
		QStringList hostList = list[2].split(",");
		if (hostList.size()==2) {
			hostname = hostList[0];
			ip = hostList[1];
		} else if (hostList.size()==1) {
			if (isIp(hostList[0])) {
				ip = hostList[0];
			} else {
				hostname = hostList[0];
			}
		} else {
			valid = false;
		}
	}

	KEY_ALGORITHM keyAlgo;
	valid = valid && list[3].size()>1 && list[3][0]==QChar('(') && list[3][list[3].count()-1]==QChar(')');
	if (valid) { // extract key algorithm
		QString keyAlgoStr = list[3].section("(", 1, -1).section(")", 0, -2);
		keyAlgo = getKeyAlgoFromString(keyAlgoStr);
		valid = keyAlgo!=KEY_ALGORITHM::ERROR;
	}

	if (valid) {
		out = std::make_shared<SshFingerprint>(hostname, ip, hash, keyAlgo, algoBitsSize);
	}

	return out;
}

SshFingerprintAndKey SshFingerprint::getFingerprintFromKnownHosts(const QString& hostname) {
	SshFingerprintAndKey out;

	QTemporaryFile file;
	file.setAutoRemove(true);
	bool fileValid = file.open();
	file.close();

	if (fileValid) {
		QProcess process;
		process.setStandardOutputFile(file.fileName());
		process.setStandardErrorFile("/dev/null");
		process.start("ssh-keygen", QStringList() << "-F" << hostname);
		bool processFinished = process.waitForFinished(-1);

		process.setStandardOutputFile("");
		process.setStandardErrorFile("");
		process.setReadChannel(QProcess::StandardOutput);

		if (processFinished) {
			process.start("ssh-keygen", QStringList() << "-l" << "-f" << file.fileName());
			processFinished = process.waitForFinished(-1);
		}
		if (processFinished) {
			bool fileOpened = file.open();
			if (fileOpened) {
				QTextStream textStream(&file);
				QString hostKey;
				while (out.hostKey==nullptr && textStream.readLineInto(&hostKey)) {
					out.hostKey = SshHostKey::getHostKey(hostKey);
				}
			}
			file.close();

			if (out.hostKey!=nullptr) {
				QByteArray standardOuput = process.readAllStandardOutput();
				QString standardOuputStr(standardOuput);
				QStringList standardOuputStrList = standardOuputStr.split("\n");
				int i = 0;
				while (out.fingerprint==nullptr && i<standardOuputStrList.count()) {
					out.fingerprint = SshFingerprint::getFingerprintFromKeyGen(standardOuputStrList[i]);
					i++;
				}
			}
		}
	}
	return out;
}

SshFingerprintAndKey SshFingerprint::getFingerprintFromCustomKnownHosts(const QString& hostname,
			const QStringList& knownHostFiles) {
	SshFingerprintAndKey out;

	QTemporaryFile file;
	file.setAutoRemove(true);
	bool fileValid = file.open();
	file.close();

	if (fileValid) {
		int knownHostsIdx = 0;
		while (out.fingerprint==nullptr && out.hostKey==nullptr && knownHostsIdx<knownHostFiles.size()) {
			QProcess process;
			process.setStandardOutputFile(file.fileName());
			process.setStandardErrorFile("/dev/null");
			process.start("ssh-keygen", QStringList() << "-F" << hostname << "-f" << knownHostFiles[knownHostsIdx]);
			bool processFinished = process.waitForFinished(-1);

			process.setStandardOutputFile("");
			process.setStandardErrorFile("");
			process.setReadChannel(QProcess::StandardOutput);

			if (processFinished) {
				process.start("ssh-keygen", QStringList() << "-l" << "-f" << file.fileName());
				processFinished = process.waitForFinished(-1);
			}
			if (processFinished) {
				bool fileOpened = file.open();
				if (fileOpened) {
					QTextStream textStream(&file);
					QString hostKey;
					while (out.hostKey==nullptr && textStream.readLineInto(&hostKey)) {
						out.hostKey = SshHostKey::getHostKey(hostKey);
					}
				}
				file.close();

				if (out.hostKey!=nullptr) {
					QByteArray standardOuput = process.readAllStandardOutput();
					QString standardOuputStr(standardOuput);
					QStringList standardOuputStrList = standardOuputStr.split("\n");
					int i = 0;
					while (out.fingerprint==nullptr && i<standardOuputStrList.count()) {
						out.fingerprint = SshFingerprint::getFingerprintFromKeyGen(standardOuputStrList[i]);

						i++;
					}
				}
			}
			knownHostsIdx++;
		}
	}

	return out;
}

QString repeateStrWithSep(const QString& str, const QString& sep, int count) {
	QStringList list;
	for (int i=0; i<count; i++) {
		list << str;
	}
	return list.join(sep);
}

bool SshFingerprint::isIp(const QString& ip) {
	QString charRange = "(0?0?[0-9]|0?[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])"; // match 0 or 000 to 255
	QString ipV4Pattern = "^" + repeateStrWithSep(charRange, ".", 4) + "$";

	QRegularExpression regexV4(ipV4Pattern);
	QRegularExpressionMatch matchV4 = regexV4.match(ip);

	bool result = matchV4.hasMatch();

	if (!result) {
		// try ip V6
		QString defaultChar = "[a-f0-9]";
		QString ipV6CharRange = "(" + defaultChar + "|" + defaultChar + defaultChar +
				"|" + repeateStrWithSep(defaultChar, "", 3) + "|" +
				repeateStrWithSep(defaultChar, "", 4) + ")";
		QString ipV6Pattern = "^" + repeateStrWithSep(ipV6CharRange, ":", 8) + "$";

		QRegularExpression regexV6(ipV6Pattern);
		QRegularExpressionMatch matchV6 = regexV6.match(ip);

		result = matchV6.hasMatch();
	}
	return result;
}

SshFingerprint::KEY_ALGORITHM SshFingerprint::getKeyAlgoFromString(const QString& _keyAlgoStr) {
	QString keyAlgoStr = _keyAlgoStr.toUpper();

	KEY_ALGORITHM keyAlgo = KEY_ALGORITHM::ERROR;
	if (keyAlgoStr.compare("DSA")) {
		keyAlgo = KEY_ALGORITHM::DSA;
	} else if (keyAlgoStr.compare("ECDSA")) {
		keyAlgo = KEY_ALGORITHM::ECDSA;
	} else if (keyAlgoStr.compare("RSA")) {
		keyAlgo = KEY_ALGORITHM::RSA;
	} else if (keyAlgoStr.compare("ED25519")) {
		keyAlgo = KEY_ALGORITHM::ED25519;
	}

	return keyAlgo;
}

QString SshFingerprint::getStringFromKeyAlgo(const KEY_ALGORITHM& key) {
	QString str;
	switch (key) {
	case KEY_ALGORITHM::DSA:
		str = "DSA";
		break;
	case KEY_ALGORITHM::ECDSA:
		str = "ECDSA";
		break;
	case KEY_ALGORITHM::RSA:
		str = "RSA";
		break;
	case KEY_ALGORITHM::ED25519:
		str = "ED25519";
		break;
	default:
		str = "ERROR";
	}
	return str;
}

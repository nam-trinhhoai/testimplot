#ifndef SSHFINGERPRINT_H
#define SSHFINGERPRINT_H

#include <QString>

#include <utility>
#include <memory>

class SshFingerprint;
class SshHostKey;

typedef struct SshFingerprintAndKey {
	std::shared_ptr<SshFingerprint> fingerprint;
	std::shared_ptr<SshHostKey> hostKey;
} SshFingerprintAndKey;

class SshFingerprint {
public:
	friend bool operator==(const SshFingerprint& a, const SshFingerprint& b);

	enum class KEY_ALGORITHM {
		DSA, ECDSA, ED25519, RSA, ERROR
	};

	SshFingerprint(const QString& hostname, const QString& ip, const QString& hash,
			KEY_ALGORITHM keyAlgorithm, int algoBitsSize);
	SshFingerprint(const SshFingerprint& fingerprint);
	~SshFingerprint();

	const SshFingerprint& operator=(const SshFingerprint& other);
	operator QString() const;

	const QString& hostname() const;
	const QString& ip() const;
	const QString& hash() const;
	KEY_ALGORITHM keyAlgorithm() const;
	int algoBitsSize() const;

	/*
	 * Identify finger print from output of "ssh-keygen -l -f"
	 * example : algoBitsSize SHA256:monsuperhash myHost,myIp (keyAlgorithm)
	 */
	static std::shared_ptr<SshFingerprint> getFingerprintFromKeyGen(const QString& fingerprint);
	static SshFingerprintAndKey getFingerprintFromKnownHosts(const QString& hostname);
	static SshFingerprintAndKey getFingerprintFromCustomKnownHosts(const QString& hostname,
			const QStringList& knownHostFiles);

	static bool isIp(const QString& ip);

	static KEY_ALGORITHM getKeyAlgoFromString(const QString& str);
	static QString getStringFromKeyAlgo(const KEY_ALGORITHM& key);

private:
	QString m_hostname;
	QString m_ip;
	QString m_hash;
	KEY_ALGORITHM m_keyAlgorithm;
	int m_algoBitsSize;
};

bool operator==(const SshFingerprint& a, const SshFingerprint& b);

//class HostValidator {
//	static bool isHostInKnownHosts(const QString& hostname);
//	static bool isFingerprintMatchingKnownHosts(const QString& hostname, const QString& fingerprint);
//	static bool compareFingerprint(const QString& fingerprint1, const QString& fingerprint2);
//	static bool addFingerprintToKnownHosts(const QString& hostname, const QString& ip, const QString& fingerprint);
//};

#endif

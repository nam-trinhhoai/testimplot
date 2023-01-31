#ifndef SRC_BNNI_PARALLELIZATION_SSHHOSTKEY_H
#define SRC_BNNI_PARALLELIZATION_SSHHOSTKEY_H

#include <QString>

#include <memory>

class SshHostKey {
public:
	friend bool operator==(const SshHostKey& a, const SshHostKey& b);

	// key from known host or output of ssh-keyscan myhost
	SshHostKey(const QString& hostKey, const QString& keyAlgorithm, const QString& key);
	SshHostKey(const SshHostKey& other);

	const QString& completeHostKey() const;
	const QString& keyAlgorithm() const;
	const QString& key() const;

	const SshHostKey& operator=(const SshHostKey&);

	static std::shared_ptr<SshHostKey> getHostKey(const QString& hostKey);

	bool addToKnownHosts() const;

private:
	QString m_completeHostKey;
	QString m_keyAlgorithm;
	QString m_key;
};

bool operator==(const SshHostKey& a, const SshHostKey& b);

#endif

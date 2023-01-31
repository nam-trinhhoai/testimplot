#ifndef SRC_BNNI_PARALLELIZATION_KERBEROSAUTHENTIFICATION_H
#define SRC_BNNI_PARALLELIZATION_KERBEROSAUTHENTIFICATION_H

class QWidget;

class KerberosAuthentification {
public:
	KerberosAuthentification();
	~KerberosAuthentification();

	bool isAuthentificated() const;
	bool authentificate(QWidget* parent);// parent for qdialog

private:
	void checkKerberosTickets();

	bool m_isAuthentificated;
};

#endif

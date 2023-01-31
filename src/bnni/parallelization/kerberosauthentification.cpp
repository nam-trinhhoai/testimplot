#include "kerberosauthentification.h"

#include <QWidget>
#include <QInputDialog>
#include <QMessageBox>
#include <QProcess>
#include <QRegularExpression>

KerberosAuthentification::KerberosAuthentification() {
	m_isAuthentificated = false;

	checkKerberosTickets();
}

KerberosAuthentification::~KerberosAuthentification() {
}

bool KerberosAuthentification::isAuthentificated() const {
	return m_isAuthentificated;
}

bool KerberosAuthentification::authentificate(QWidget* parent) {
	checkKerberosTickets();

	bool valid = !m_isAuthentificated;
	QString user;
	if (valid) {
		// get user
		QProcess processGetUser;
		processGetUser.start("whoami");
		processGetUser.waitForFinished();

		valid = processGetUser.exitCode()==QProcess::NormalExit;

		user = processGetUser.readAllStandardOutput();
		user = user.trimmed();
	}

	QString password;
	if (valid) {
		password = QInputDialog::getText(parent, "Linux Password for kinit", "Linux Password", QLineEdit::Password);
		valid = !password.isNull() && !password.isEmpty();
	}

	if (valid) {
		QProcess process;
		process.start("kinit", QStringList() << "x" + user);
		process.write(password.toUtf8());
		process.closeWriteChannel();
		process.waitForFinished(-1);

		QString errorStr(process.readAllStandardError());
		QRegularExpression regexp("kinit\\: Password incorrect while getting initial credentials");
		QRegularExpressionMatch match = regexp.match(errorStr);
		valid = !match.hasMatch();

		if (valid) {
			checkKerberosTickets();
		} else {
			QMessageBox::warning(parent, "Bad password", "kinit rejected the password");
		}
	}

	return m_isAuthentificated;
}

void KerberosAuthentification::checkKerberosTickets() {
	QProcess process;
	process.start("klist");
	process.waitForFinished(-1);

	QString errorStr(process.readAllStandardError());
	QRegularExpression regexp("klist\\: No credentials cache found \\(filename\\: .*\\)");
	QRegularExpressionMatch match = regexp.match(errorStr);

	m_isAuthentificated = !match.hasMatch();
}

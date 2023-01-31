#include "processwatcherwidget.h"

#include <QDebug>
#include <QProcess>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>


ProcessWatcherWidget::ProcessWatcherWidget(QWidget* parent) : QWidget(parent) {
	m_textEdit = new QTextEdit;
	m_textEdit->setReadOnly(true);

	QVBoxLayout* layout = new QVBoxLayout;
	setLayout(layout);
	layout->addWidget(m_textEdit);

	QPushButton* killButton = new QPushButton("Kill process");
	layout->addWidget(killButton);
	QPushButton* terminateButton = new QPushButton("Terminate process");
	layout->addWidget(terminateButton);

	connect(killButton, &QPushButton::clicked, this, &ProcessWatcherWidget::processKill);
	connect(terminateButton, &QPushButton::clicked, this, &ProcessWatcherWidget::processTerminate);

	m_currentProcessId = c_INVALID_ID;
}

ProcessWatcherWidget::~ProcessWatcherWidget() {

}

std::size_t ProcessWatcherWidget::launchProcess(const QString& cmd, const QStringList& options,
			const QString& workingDir, const QProcessEnvironment& env) {
	if (m_currentProcessId!=c_INVALID_ID && processState()!=QProcess::ProcessState::NotRunning) {
		return c_INVALID_ID;
	}

	reset();

	m_process.reset(new QProcess);

	m_process->setProcessChannelMode(QProcess::MergedChannels);
	m_process->setReadChannel(QProcess::StandardOutput);

	m_process->setProgram(cmd);
	m_process->setArguments(options);
	m_process->setWorkingDirectory(workingDir);
	m_process->setProcessEnvironment(env);

	QProcess* processPtr = m_process.get();
	connect(processPtr, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &ProcessWatcherWidget::processFinished);
	connect(processPtr, &QProcess::errorOccurred, this, &ProcessWatcherWidget::errorOccured);
	connect(processPtr, &QProcess::readyRead, this, &ProcessWatcherWidget::updateTextEdit);

	m_process->start();
	m_process->waitForStarted();
	qDebug() << m_process->state();
}

int ProcessWatcherWidget::processExitCode() const {
	int out;
	if (m_process!=nullptr) {
		out = m_process->exitCode();
	} else {
		out = 0;
	}
	return out;
}

QProcess::ExitStatus ProcessWatcherWidget::processExitStatus() const {
	QProcess::ExitStatus out;
	if (m_process!=nullptr) {
		out = m_process->exitStatus();
	} else {
		out = QProcess::ExitStatus::NormalExit;
	}
	return out;
}

QProcess::ProcessError ProcessWatcherWidget::processError() const {
	QProcess::ProcessError out;
	if (m_process!=nullptr) {
		out = m_process->error();
	} else {
		out = QProcess::ProcessError::UnknownError;
	}
	return out;
}

QProcess::ProcessState ProcessWatcherWidget::processState() const {
	QProcess::ProcessState out;
	if (m_process!=nullptr) {
		out = m_process->state();
	} else {
		out = QProcess::ProcessState::NotRunning;
	}
	return out;
}

void ProcessWatcherWidget::processKill() {
	if (m_process!=nullptr) {
		m_process->kill();
	}
}

void ProcessWatcherWidget::processTerminate() {
	if (m_process!=nullptr) {
		m_process->terminate();
	}
}

void ProcessWatcherWidget::reset() {
	if (m_process!=nullptr) {
		// disconnect process signals
		QProcess* processPtr = m_process.get();
		disconnect(processPtr, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &ProcessWatcherWidget::processFinished);
		disconnect(processPtr, &QProcess::errorOccurred, this, &ProcessWatcherWidget::errorOccured);
		disconnect(processPtr, &QProcess::readyRead, this, &ProcessWatcherWidget::updateTextEdit);

		m_process.reset(nullptr);
	}
	m_currentProcessId = c_INVALID_ID;
	m_textEdit->clear();
}

void ProcessWatcherWidget::updateTextEdit() {
	if (m_process==nullptr) {
		return;
	}
	QByteArray data = m_process->readAll();

	QString newData(data);
	m_textEdit->append(newData);
}

void ProcessWatcherWidget::processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
	emit processEnded(m_currentProcessId, exitCode, exitStatus);
}

void ProcessWatcherWidget::errorOccured(QProcess::ProcessError error) {
	emit processGotError(m_currentProcessId, error);
}

std::size_t ProcessWatcherWidget::INVALID_ID() {
	return ProcessWatcherWidget::c_INVALID_ID;
}

std::size_t ProcessWatcherWidget::c_INVALID_ID = 0.0;

#ifndef SRC_WIDGET_PROCESSWATCHERWIDGET_H_
#define SRC_WIDGET_PROCESSWATCHERWIDGET_H_

#include <QWidget>
#include <QProcessEnvironment>

#include <memory>

class QProcess;
class QTextEdit;

class ProcessWatcherWidget : public QWidget {
	Q_OBJECT
public:
	ProcessWatcherWidget(QWidget* parent=0);
	~ProcessWatcherWidget();

	std::size_t launchProcess(const QString& cmd, const QStringList& options,
			const QString& workingDir=QString(),
			const QProcessEnvironment& env=QProcessEnvironment::systemEnvironment());

	int processExitCode() const;
	QProcess::ExitStatus processExitStatus() const;
	QProcess::ProcessError processError() const;
	QProcess::ProcessState processState() const;

	void processKill();
	void processTerminate();

	static std::size_t INVALID_ID();

signals:
	void processEnded(std::size_t processId, int exitCode, QProcess::ExitStatus exitStatus);
	void processGotError(std::size_t processId, QProcess::ProcessError error);

private slots:
	void reset();
	void updateTextEdit();
	void processFinished(int exitCode, QProcess::ExitStatus exitStatus);
	void errorOccured(QProcess::ProcessError error);

private:
	QTextEdit* m_textEdit;

	std::size_t m_currentProcessId;
	std::unique_ptr<QProcess> m_process;

	std::size_t m_nextId = 1;
	static std::size_t c_INVALID_ID;
};


#endif

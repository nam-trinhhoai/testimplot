#ifndef SRC_INFORMATIONMANAGER_VIDEOS_VIDEOMANAGERWRAPPER_H
#define SRC_INFORMATIONMANAGER_VIDEOS_VIDEOMANAGERWRAPPER_H

#include "managerwidget.h"

#include <QPointer>
#include <QWidget>

class ProjectManager;
class ProjectManagerWidget;
class SurveyManager;

class QTabWidget;

class VideoManagerWrapper : public QWidget {
	Q_OBJECT
public:
	VideoManagerWrapper(ProjectManagerWidget *projectmanager, QWidget* parent = nullptr);
	~VideoManagerWrapper();

public slots:
	void loadSessionGui();
	void projectChanged();
	void surveyChanged();

private:
	void loadLastSession();
	void loadSession(const QString& sessionPath);

	QPointer<ManagerWidget> m_managerWidget = nullptr;
	ProjectManager* m_projectManager = nullptr;
	ProjectManagerWidget* m_projectManagerWidget = nullptr;
	SurveyManager* m_surveyManager = nullptr;
	QTabWidget* m_tabWidget;

	QString m_cacheProjectPath;
	QString m_cacheSurveyPath;
};

#endif // SRC_INFORMATIONMANAGER_VIDEOS_VIDEOMANAGERWRAPPER_H

#ifndef SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETMANAGERWRAPPER_H
#define SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETMANAGERWRAPPER_H

#include "managerwidget.h"

#include <QPointer>
#include <QWidget>

class ProjectManager;

class QTabWidget;

class TrainingSetManagerWrapper : public QWidget {
	Q_OBJECT
public:
	TrainingSetManagerWrapper(QWidget* parent = nullptr);
	~TrainingSetManagerWrapper();

public slots:
	void loadSessionGui();
	void projectChanged();

private:
	void loadLastSession();
	void loadSession(const QString& sessionPath);

	QPointer<ManagerWidget> m_managerWidget = nullptr;
	ProjectManager* m_projectManager;
	QTabWidget* m_tabWidget;

	QString m_cacheProjectPath;
};

#endif // SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETMANAGERWRAPPER_H

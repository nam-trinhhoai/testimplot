#include "trainingsetmanagerwrapper.h"

#include "globalconfig.h"
#include "ProjectManager.h"
#include "ProjectManagerWidget.h"
#include "trainingsetinformationaggregator.h"

#include <QFileDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPushButton>
#include <QSettings>
#include <QSizeGrip>
#include <QTabWidget>
#include <QVBoxLayout>

TrainingSetManagerWrapper::TrainingSetManagerWrapper(QWidget* parent) :
		QWidget(parent) {
	setAttribute(Qt::WA_DeleteOnClose);
	setWindowTitle("Trainingset Information");
	setContentsMargins(0, 0, 0, 0);

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_tabWidget = new QTabWidget;
	mainLayout->addWidget(m_tabWidget);

	QWidget* holder = new QWidget;
	m_tabWidget->addTab(holder, "Manager");
	QVBoxLayout* holderLayout = new QVBoxLayout;
	holder->setLayout(holderLayout);

	QPushButton* loadSessionButton = new QPushButton("Load session");
	holderLayout->addWidget(loadSessionButton);

	m_projectManager = new ProjectManager;
	holderLayout->addWidget(m_projectManager);

	QSizeGrip* sizeGrip = new QSizeGrip(this);
	sizeGrip->setContentsMargins(0, 0, 0, 0);
	mainLayout->addWidget(sizeGrip, 0, Qt::AlignRight);

	connect(loadSessionButton, &QPushButton::clicked, this, &TrainingSetManagerWrapper::loadSessionGui);
	connect(m_projectManager, &ProjectManager::projectChanged, this, &TrainingSetManagerWrapper::projectChanged);

	// auto load last session
	loadLastSession();
}

TrainingSetManagerWrapper::~TrainingSetManagerWrapper() {

}

void TrainingSetManagerWrapper::loadSessionGui() {
    QSettings settings;
    GlobalConfig& config = GlobalConfig::getConfig();
    const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, config.sessionPath()).toString();

    const QString filePath = QFileDialog::getOpenFileName(this,
                                                          tr("Load Session"),
                                                          lastPath,
                                                          QLatin1String("*.json"));
    if (filePath.isEmpty())
        return;

    QFileInfo fi(filePath);
    settings.setValue(LAST_SESSION_PATH_IN_SETTINGS, fi.absoluteFilePath());

    loadSession(filePath);
}

void TrainingSetManagerWrapper::projectChanged() {
	if (m_cacheProjectPath.compare(m_projectManager->getPath())==0) {
		return;
	}
	QString projectName = m_projectManager->getName();
	if (projectName.isNull() || projectName.isEmpty()) {
		m_cacheProjectPath = "";
	} else {
		m_cacheProjectPath = m_projectManager->getPath();
	}

	if (m_managerWidget!=nullptr && m_tabWidget->indexOf(m_managerWidget)>=0) {
		int idx = m_tabWidget->indexOf(m_managerWidget);
		QWidget* widget = m_tabWidget->widget(idx);
		m_tabWidget->removeTab(idx);
		widget->deleteLater();
		m_managerWidget = nullptr;
	}
	if (!m_cacheProjectPath.isNull() || !m_cacheProjectPath.isEmpty()) {
		TrainingSetInformationAggregator* aggregator = new TrainingSetInformationAggregator(m_cacheProjectPath);
		m_managerWidget = new ManagerWidget(aggregator);
		m_tabWidget->insertTab(0, m_managerWidget, "Information");
	}
}

void TrainingSetManagerWrapper::loadLastSession() {
	QSettings settings;
	const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, "").toString();
	if (lastPath.isEmpty() || !QFileInfo(lastPath).exists())
		return;

	loadSession(lastPath);
}

void TrainingSetManagerWrapper::loadSession(const QString& sessionPath) {
	QFile file(sessionPath);
	if (!file.open(QIODevice::ReadOnly))
	{
		qDebug() << "DataManager : cannot load session, file not readable";
		return;
	}

	QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
	if (!doc.isObject())
	{
		qDebug() << "DataManager : cannot load session, root is not a json object";
		return;
	}

	bool isProjectValid = false;
	bool isSurveyValid = false;

	QJsonObject rootObj = doc.object();
	QString project = "";
	QString survey = "";

	if (rootObj.contains(projectTypeKey0))
	{
		QString projectPath = "";
		QString project_type = rootObj.value(projectTypeKey0).toString("None");
		if ( project_type.compare("USER") == 0 && rootObj.contains(projectPathKey0) )
		{
			projectPath = rootObj.value(projectPathKey0).toString(".");
			m_projectManager->setUserProjectPath(projectPath);
		}
		bool isProjectTypeValid = m_projectManager->setProjectType(project_type);
		if (isProjectTypeValid && rootObj.contains(projectKey0))
		{ // load project
			QString project = rootObj.value(projectKey0).toString("");

			if (m_projectManager->setProjectName(project))
			{
				isProjectValid = true;
			}
		}
	}
}

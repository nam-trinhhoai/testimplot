#include "videomanagerwrapper.h"

#include "globalconfig.h"
#include "ProjectManager.h"
#include "ProjectManagerWidget.h"
#include "SurveyManager.h"
#include <ProjectManagerWidget.h>
#include "videoinformationaggregator.h"

#include <QFileDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPushButton>
#include <QSettings>
#include <QSizeGrip>
#include <QTabWidget>
#include <QVBoxLayout>

VideoManagerWrapper::VideoManagerWrapper(ProjectManagerWidget *projectmanager, QWidget* parent) :
		QWidget(parent) {
	setAttribute(Qt::WA_DeleteOnClose);
	setWindowTitle("Video Information");
	setContentsMargins(0, 0, 0, 0);

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_tabWidget = new QTabWidget;
	m_tabWidget->setIconSize(QSize(40, 40));
	mainLayout->addWidget(m_tabWidget);

	QWidget* holder = new QWidget;
	m_tabWidget->addTab(holder, QIcon(":/slicer/icons/earth.png"), "Manager");
	QVBoxLayout* holderLayout = new QVBoxLayout;
	holder->setLayout(holderLayout);

	QPushButton* loadSessionButton = new QPushButton("Load session");
	holderLayout->addWidget(loadSessionButton);

	QHBoxLayout* managerLayout = new QHBoxLayout;
	holderLayout->addLayout(managerLayout);

	m_projectManager = new ProjectManager;
	managerLayout->addWidget(m_projectManager);

	m_surveyManager = new SurveyManager;
	m_projectManager->setSurveyManager(m_surveyManager);
	managerLayout->addWidget(m_surveyManager);

	QSizeGrip* sizeGrip = new QSizeGrip(this);
	sizeGrip->setContentsMargins(0, 0, 0, 0);
	mainLayout->addWidget(sizeGrip, 0, Qt::AlignRight);

	connect(loadSessionButton, &QPushButton::clicked, this, &VideoManagerWrapper::loadSessionGui);
	connect(m_projectManager, &ProjectManager::projectChanged, this, &VideoManagerWrapper::projectChanged);
	connect(m_surveyManager, &SurveyManager::surveyChanged, this, &VideoManagerWrapper::surveyChanged);

	// auto load last session
	loadLastSession();
}

VideoManagerWrapper::~VideoManagerWrapper() {

}

void VideoManagerWrapper::loadSessionGui() {
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

void VideoManagerWrapper::projectChanged() {
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
}

void VideoManagerWrapper::loadLastSession() {
	QSettings settings;
	const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, "").toString();
	if (lastPath.isEmpty() || !QFileInfo(lastPath).exists())
		return;

	loadSession(lastPath);
}

void VideoManagerWrapper::loadSession(const QString& sessionPath) {
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
		if (isProjectValid && rootObj.contains(surveyKey0))
		{
			QString survey = rootObj.value(surveyKey0).toString("");
			if (m_surveyManager->setForceName(survey))
			{
				isSurveyValid = true;
			}
		}
	}
}

void VideoManagerWrapper::surveyChanged() {
	if (m_cacheSurveyPath.compare(m_surveyManager->getPath())==0) {
		return;
	}
	QString surveyName = m_surveyManager->getName();
	if (surveyName.isNull() || surveyName.isEmpty()) {
		m_cacheSurveyPath = "";
	} else {
		m_cacheSurveyPath = m_surveyManager->getPath();
	}

	if (m_managerWidget!=nullptr && m_tabWidget->indexOf(m_managerWidget)>=0) {
		int idx = m_tabWidget->indexOf(m_managerWidget);
		QWidget* widget = m_tabWidget->widget(idx);
		m_tabWidget->removeTab(idx);
		widget->deleteLater();
		m_managerWidget = nullptr;
	}
	if (!m_cacheSurveyPath.isNull() || !m_cacheSurveyPath.isEmpty()) {
		VideoInformationAggregator* aggregator = new VideoInformationAggregator(m_cacheSurveyPath);
		m_managerWidget = new ManagerWidget(aggregator);
		m_tabWidget->insertTab(0, m_managerWidget, QIcon(":/slicer/icons/mainwindow/VideoPlayer.svg"), "Information");
	}
}

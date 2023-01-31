/*
 *
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */

#ifndef __PROJECTMANAGER__
#define __PROJECTMANAGER__

#include <vector>

#include <QWidget>
#include <QString>
#include <QStringList>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QListWidget>
#include <QDir>
#include <QLineEdit>
#include <QTabWidget>
#include <QGroupBox>
#include <QTableWidget>
#include <QPushButton>
#include <QVBoxLayout>


#include <vector>
#include <math.h>
#include <ProjectManagerNames.h>
#include <SurveyManager.h>
#include <CulturalsManager.h>
#include <WellsManager.h>
#include <picksmanager.h>


class ProjectManager : public QWidget{
	Q_OBJECT
public:
	ProjectManager(QWidget* parent = 0);
	virtual ~ProjectManager();
	QListWidget *listwidgetProject;
	void setSurveyManager(SurveyManager *surveyManager);
	void setCulturalsManager(CulturalsManager *culturalsManager);
	void setWellsManager(WellsManager *wellsManager);
	void setPicksManager(PicksManager *picksManager);
	QComboBox *comboboxProjectType;
	bool setProjectType(int idx);
	bool setProjectType(QString projectDirName);
	bool setProjectName(QString name);

	QString getType();
	QString getName();
	QString getPath();
	void setUserProjectPath(QString path);

	static std::vector<QString> getListDir(QString path);

signals:
	void projectChanged();

private:
	int project_path_nbre = 0;
	QStringList arrayProjectTypeNames;
	QStringList arrayProjectPath;

	SurveyManager *m_surveyManager;
	CulturalsManager *m_culturalsManager;
	WellsManager *m_wellsManager;
	PicksManager *m_picksManager = nullptr;
	QLineEdit *lineeditUserProjectPath, *lineeditProjectSearch;
	QPushButton *pushbuttonUserProjectPathValid;
	QLabel *labelUserProjectPath;

	void setCallBackListClick(void (*ptr)(QListWidgetItem*));
	QString getRootProjectPath(int idx);
	QString getProjectFullPath();
	std::vector<std::vector<QString>> projectListTinyName, projectListFullName;

	ProjectManagerNames *names = nullptr;
	std::vector<ProjectManagerNames*> cacheNames;

	void updateNames(int idx);
	void displayNames();

	QString getProjectType();
	QString getProjectPath();
	QString getProjectName();
	void projectTypeEnable();



public slots:
	void trt_projecttypeclick(int idx);
	void trt_projetlistClick(QListWidgetItem*);
	void trt_projectSearChchange(QString str);
	void trt_userProjectValid();


};


#endif

/*
 *
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */

#ifndef __PROJECTMANAGERWIDGET__
#define __PROJECTMANAGERWIDGET__

#include <vector>

#include <QWidget>
#include <QString>
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
#include <QGroupBox>

#include <vector>
#include <math.h>

#include <ProjectManager.h>
#include <SurveyManager.h>
#include <SeismicManager.h>
#include <HorizonManager.h>
#include <CulturalsManager.h>
#include <WellsManager.h>
#include <picksmanager.h>
#include <WellUtil.h>
#include <RgbRawManager.h>



class ProjectManagerWidget : public QWidget{
	Q_OBJECT
public:
	enum ManagerTabName {
		SEISMIC, HORIZON, CULTURAL, RGBRAW, WELL, PICK
	};

	ProjectManagerWidget(bool sessionVisible = true, QWidget* parent = 0);
	virtual ~ProjectManagerWidget();
	QString getProjectType();
	QString getProjectName();
	QString getProjectPath();
	QString getSurveyName();
	QString getSurveyPath();
	QString getImportExportPath();
	QString getIJKPath();
	// QString getPatchPath();
	QString getSeismicDirectory();
	QString getHorizonsPath();
	QString getHorizonsIsoValPath();
	QString getHorizonsFreePath();

	QString getNextVisionPath();
	QString getNVHorizonPath();
	QString getIsoHorizonPath();
	QString getNextVisionSeismicPath();
	QString getPatchPath();
	QString getVideoPath();

	bool mkdirNextVisionPath();
	bool mkdirNVHorizonPath();
	bool mkdirIsoHorizonPath();
	bool mkdirNextVisionSeismicPath();
	bool mkdirPatchPath();
	bool mkdirVideoPath();
	void createPatchDir();

	std::vector<QString> getSeismicNames();
	std::vector<QString> getSeismicPath();
	std::vector<QString> getSeismicAllNames();
	std::vector<QString> getSeismicAllPath();
	std::vector<int> getSeismicAllDimx();
	std::vector<int> getSeismicAllDimy();
	std::vector<int> getSeismicAllDimz();

	std::vector<QString> getHorizonNames();
	std::vector<QString> getHorizonPath();
	std::vector<QString> getHorizonAllNames();
	std::vector<QString> getHorizonAllPath();

	std::vector<QString> getCulturalsCdatNames();
	std::vector<QString> getCulturalsCdatPath();
	std::vector<QString> getCulturalsStrdNames();
	std::vector<QString> getCulturalsStrdPath();
	WELLMASTER get_well_list();
	std::vector<QString> getRgbRawNames();
	std::vector<QString> getRgbRawPath();
	std::vector<QString> getRgbRawDirectoryNames();
	std::vector<QString> getRgbRawDirectoryPath();
	void RgbRawUpdateNames();

	std::vector<QString> getHorizonIsoValueListName();
	std::vector<QString> getHorizonIsoValueListPath();

	std::vector<QString> getFreeHorizonNames();
	std::vector<QString> getFreeHorizonFullName();

	std::vector<QString> getAviNames();
	std::vector<QString> getAviPath();

	// compatibility version 0
	std::vector<QString> get_seismic_names();
	QString horizon_path_read();
	std::vector<QString> get_seismic_fullpath_names();
	std::vector<QString> get_horizon_names();
	std::vector<QString> get_horizon_fullpath_names();

	void onlyShow(const std::vector<ManagerTabName>& tabNames);
	void showTab(ManagerTabName tabName);
	void hideTab(ManagerTabName tabName);
	/*
	 * type = log, picks or tf2p
	 * nameType = tiny or full
	 */

	std::vector<std::vector<QString>> getWellBasketLogPicksTf2pNames(QString type, QString nameType);
	std::vector<QString> getWellBasketTinyNames();
	std::vector<QString> getWellBasketFullNames();
	std::vector<QBrush> getPicksBasketColors();

	std::vector<QString> getPicksBasketFullNames();
	std::vector<QString> getPicksBasketNames();

	std::vector<WELLHEADDATA> getMainWellData();
	std::vector<MARKER> getPicksSortedWells();


public:
	ProjectManager *m_projectManager;
	SurveyManager *m_surveyManager;
	SeismicManager *m_seismicManager;
	HorizonManager *m_horizonManager;
	CulturalsManager *m_culturalsManager;
	WellsManager *m_wellsManager;
	PicksManager *m_picksManager = nullptr;
	RgbRawManager *m_rgbRawManager;
	QTabWidget *tabwidget;
	void load_session(QString sessionPath);
	void save_session(QString sessionPath);
	void load_session_gui();
	void save_session_gui();
	void load_last_session();
	void fill_empty_logs_list();
	void seimsicDatabaseUpdate();
	void wellDatabaseUpdate();



public slots:
	void trt_projetlistClick(QListWidgetItem*);
	void trt_session_save();
	void trt_session_load();
	void trt_debug();

private:
	QWidget* widgetFromTabName(ManagerTabName tabName);

	QGroupBox* m_qgb_seismic, *m_qgb_horizon, *m_qgb_culturals, *m_qgb_wells;
	QGroupBox* m_qgb_picks;
	QGroupBox* m_qgb_rgbRaw;
	std::map<ManagerTabName, int> m_nameToIndex;
	std::map<ManagerTabName, bool> m_isPageShownMap;
};

static const QString projectTypeKey0 = QStringLiteral("projectType");
static const QString projectPathKey0 = QStringLiteral("projectPath");
static const QString projectKey0 = QStringLiteral("project");
static const QString surveyKey0 = QStringLiteral("survey");
static const QString seismicKey0 = QStringLiteral("seismic");
static const QString horizonKey0 = QStringLiteral("horizon");
static const QString culturalKey0 = QStringLiteral("cultural");
static const QString wellKey0 = QStringLiteral("well");
static const QString wellPathKey0 = QStringLiteral("wellPath");
static const QString wellLogKey0 = QStringLiteral("wellLogs");
static const QString wellTFPKey0 = QStringLiteral("wellTFP");
//static const QString wellPickKey0 = QStringLiteral("wellPicks");
static const QString neuronKey0 = QStringLiteral("neuron");
static const QString picksNamesKey0 = QStringLiteral("picksNames");
static const QString picksPathKey0 = QStringLiteral("picksPath");

static const QString tinynameKey0 = QStringLiteral("tinyname");
static const QString fullnameKey0 = QStringLiteral("fullname");


#endif

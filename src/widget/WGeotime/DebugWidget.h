
#ifndef __DEBUGWIDGET__
#define __DEBUGWIDGET__


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

#include <vector>
#include <math.h>

#include <ProjectManagerWidget.h>
#include <ProjectManager.h>
#include <SurveyManager.h>
#include <SeismicManager.h>
#include <HorizonManager.h>
#include <CulturalsManager.h>
#include <WellsManager.h>
#include <RgbRawManager.h>









class DebugWidget : public QWidget{
	Q_OBJECT
public:
	DebugWidget(QWidget* parent = 0);
	virtual ~DebugWidget();

private:
	ProjectManagerWidget *m_projectManagerWidget;
	ProjectManager *m_projectManager;
	SurveyManager *m_surveyManager;
	SeismicManager *m_seismicManager;
	HorizonManager *m_horizonManager;
	CulturalsManager *m_culturalsManager;
	WellsManager *m_wellsManager;
	RgbRawManager *m_rgbRawManager;
	QTabWidget *tabwidget;
	// void load_session(QString sessionPath) ;
	WELLMASTER WellBasket;

	QString m_projectType, m_projectName, m_projectPath,
	m_surveyPath, m_surveyName;
	std::vector<QString> m_seismicName;
	std::vector<QString> m_seismicPath;
	std::vector<QString> m_horizonName;
	std::vector<QString> m_horizonPath;
	std::vector<QString> m_culturalsCdatName;
	std::vector<QString> m_culturalsCdatPath;
	std::vector<QString> m_culturalsStrdName;
	std::vector<QString> m_culturalsStrdPath;
	std::vector<QString> m_rgbrawnames;
	std::vector<QString> m_rgbrawPath;



private slots:
	void trt_testProjectManager();
	void trt_testProjectManagerGetType();
	void trt_testProjectManagerGetName();
	void trt_testProjectManagerGetPath();
	void trt_testSurveyManagerGetName();
	void trt_testSurveyManagerGetPath();
	void trt_testProjectManagerGetSeismicName();
	void trt_testProjectManagerGetHorizonName();
	void trt_testProjectManagerGetCulturalName();
	void trt_testProjectManagerGetWellName();
	void trt_testProjectManagerGetRGBName();




	void trt_testProjectManagerGetSeismicPath();

	void trt_testCallSeismicWidget();
	void trt_testCallHorizonWidget();
	void trt_testCallCulturalsWidget();
	void trt_testCallWellWidget();
	void trt_testCallRgbSpectrumWidget();



	void trt_testProjectList();
	void trt_testProjectPath();
	void trt_testSurvey();
	void trt_initSurvey();
	void trt_testSurveyPath();
	void trt_testSeismic();
	void trt_initSeismic();
	void trt_getSeismic();
	void trt_setSeismicBasket();
	void trt_testHorizon();
	void trt_initHorizon();
	void trt_getHorizon();
	void trt_setHorizonBasket();
	void trt_testCulturals();
	void trt_initCulturals();
	void trt_getCulturals();
	void trt_setCulturalsBasket();

	void trt_testWells();
	void trt_initWells();
	void trt_getWells();
	void trt_setWellsBasket();


//	void trt_projetlistClick(QListWidgetItem*);
//	void trt_session_load();

};



#endif

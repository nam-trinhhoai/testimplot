

#include <QSettings>
#include <QProcess>
#include <QFileDialog>

#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>

#include <QDebug>

#include <DebugWidget.h>

DebugWidget::DebugWidget(QWidget* parent) :
		QWidget(parent) {

	QVBoxLayout *mainLayout00 = new QVBoxLayout(this);

	m_projectManagerWidget = nullptr;


	QPushButton *pushbutton_testProjectManager = new QPushButton("Project manager");
	QPushButton *pushbutton_testProjectManagerGetType = new QPushButton("Project manager get type");
	QPushButton *pushbutton_testProjectManagerGetName = new QPushButton("Project manager get name");
	QPushButton *pushbutton_testProjectManagerGetPath = new QPushButton("Project manager get path");

	QPushButton *pushbutton_testSurveyManagerGetName = new QPushButton("Project survey get name");
	QPushButton *pushbutton_testSurveyManagerGetPath = new QPushButton("Project survey get path");

	QPushButton *pushbutton_testProjectManagerSeismicGetName = new QPushButton("Project seismic parameters");
	// QPushButton *pushbutton_testProjectmanagerSeismicGetPath = new QPushButton("Project seismic get path");

	QPushButton *pushbutton_testProjectManagerHorizonGetName = new QPushButton("Project horizon parameters");
	// QPushButton *pushbutton_testProjectmanagerHorizonGetPath = new QPushButton("Project horizon get path");

	QPushButton *pushbutton_testProjectManagerCulturalGetName = new QPushButton("Project cultural parameters");
	// QPushButton *pushbutton_testProjectmanagerCulturalGetPath = new QPushButton("Project cultural get path");

	QPushButton *pushbutton_testProjectManagerWellGet = new QPushButton("Project well get");
	QPushButton *pushbutton_testProjectManagerRgbRawGet = new QPushButton("Project RGB Spectrum get");







	QPushButton *pushbutton_testProjectList = new QPushButton("test project list");
	QPushButton *pushbutton_projectPath = new QPushButton("project path");

	QPushButton *pushbutton_testSurvey = new QPushButton("test survey");
	QPushButton *pushbutton_initSurvey = new QPushButton("init survey");
	QPushButton *pushbutton_surveyPath = new QPushButton("survey path");

	QPushButton *pushbutton_testSeismic = new QPushButton("test seismic");
	QPushButton *pushbutton_initSeismic = new QPushButton("init seismic");
	QPushButton *pushbutton_getSeismic = new QPushButton("get seimsic");
	QPushButton *pushbutton_setSeismicBasket = new QPushButton("set seimsic basket");

	QPushButton *pushbutton_testHorizon = new QPushButton("test horizon");
	QPushButton *pushbutton_initHorizon = new QPushButton("init horizon");
	QPushButton *pushbutton_getHorizon = new QPushButton("get horizon");
	QPushButton *pushbutton_setHorizonBasket = new QPushButton("set horizon basket");

	QPushButton *pushbutton_testCulturals = new QPushButton("test culturals");
	QPushButton *pushbutton_initCulturals = new QPushButton("init culturals");
	QPushButton *pushbutton_getCulturals = new QPushButton("get culturals");
	QPushButton *pushbutton_setCulturalsBasket = new QPushButton("set culturals basket");

	QPushButton *pushbutton_testWells = new QPushButton("test wells");
	QPushButton *pushbutton_initWells = new QPushButton("init wells");
	QPushButton *pushbutton_getWells = new QPushButton("get wells");
	QPushButton *pushbutton_setWellsBasket = new QPushButton("set wells basket");

	QPushButton *pushbutton_setSeismicWidget = new QPushButton("call seismic widget **** ");
	QPushButton *pushbutton_setHorizonWidget = new QPushButton("call horizon widget **** ");
	QPushButton *pushbutton_setCulturalsWidget = new QPushButton("call culturals widget **** ");
	QPushButton *pushbutton_setWellWidget = new QPushButton("call well widget **** ");
	QPushButton *pushbutton_setRgbSpectrumWidget = new QPushButton("call rgb spectrum widget **** ");



	mainLayout00->addWidget(pushbutton_testProjectManager);

	QHBoxLayout *mainLayout01 = new QHBoxLayout(this);
	mainLayout01->addWidget(pushbutton_testProjectManagerGetType);
	mainLayout01->addWidget(pushbutton_testProjectManagerGetName);
	mainLayout01->addWidget(pushbutton_testProjectManagerGetPath);

	QHBoxLayout *mainLayout02 = new QHBoxLayout(this);
	mainLayout02->addWidget(pushbutton_testSurveyManagerGetName);
	mainLayout02->addWidget(pushbutton_testSurveyManagerGetPath);

	QHBoxLayout *mainLayout03 = new QHBoxLayout(this);
	mainLayout03->addWidget(pushbutton_testProjectManagerSeismicGetName);
	// mainLayout03->addWidget(pushbutton_testProjectmanagerSeismicGetPath);

	QHBoxLayout *mainLayout04 = new QHBoxLayout(this);
	mainLayout04->addWidget(pushbutton_testProjectManagerHorizonGetName);
	// mainLayout04->addWidget(pushbutton_testProjectmanagerHorizonGetPath);

	QHBoxLayout *mainLayout05 = new QHBoxLayout(this);
	mainLayout05->addWidget(pushbutton_testProjectManagerCulturalGetName);
	// mainLayout05->addWidget(pushbutton_testProjectmanagerCulturalGetPath);

	QHBoxLayout *mainLayout06 = new QHBoxLayout(this);
	mainLayout06->addWidget(pushbutton_testProjectManagerWellGet);

	mainLayout00->addLayout(mainLayout01);
	mainLayout00->addLayout(mainLayout02);
	mainLayout00->addLayout(mainLayout03);
	mainLayout00->addLayout(mainLayout04);
	mainLayout00->addLayout(mainLayout05);
	mainLayout00->addLayout(mainLayout06);
	mainLayout00->addWidget(pushbutton_testProjectManagerRgbRawGet);
	mainLayout00->addWidget(pushbutton_setSeismicWidget);
	mainLayout00->addWidget(pushbutton_setHorizonWidget);
	mainLayout00->addWidget(pushbutton_setCulturalsWidget);
	mainLayout00->addWidget(pushbutton_setWellWidget);
	mainLayout00->addWidget(pushbutton_setRgbSpectrumWidget);



/*
	mainLayout00->addWidget(pushbutton_testProjectList);
	mainLayout00->addWidget(pushbutton_projectPath);

	mainLayout00->addWidget(pushbutton_testSurvey);
	mainLayout00->addWidget(pushbutton_initSurvey);
	mainLayout00->addWidget(pushbutton_surveyPath);

	mainLayout00->addWidget(pushbutton_testSeismic);
	mainLayout00->addWidget(pushbutton_initSeismic);
	mainLayout00->addWidget(pushbutton_getSeismic);
	mainLayout00->addWidget(pushbutton_setSeismicBasket);

	mainLayout00->addWidget(pushbutton_testHorizon);
	mainLayout00->addWidget(pushbutton_initHorizon);
	mainLayout00->addWidget(pushbutton_getHorizon);
	mainLayout00->addWidget(pushbutton_setHorizonBasket);

	mainLayout00->addWidget(pushbutton_testCulturals);
	mainLayout00->addWidget(pushbutton_initCulturals);
	mainLayout00->addWidget(pushbutton_getCulturals);
	mainLayout00->addWidget(pushbutton_setCulturalsBasket);

	mainLayout00->addWidget(pushbutton_testWells);
	mainLayout00->addWidget(pushbutton_initWells);
	mainLayout00->addWidget(pushbutton_getWells);
	mainLayout00->addWidget(pushbutton_setWellsBasket);
	*/


	connect(pushbutton_testProjectManager, SIGNAL(clicked()), this, SLOT(trt_testProjectManager()));
	connect(pushbutton_testProjectManagerGetType, SIGNAL(clicked()), this, SLOT(trt_testProjectManagerGetType()));
	connect(pushbutton_testProjectManagerGetName, SIGNAL(clicked()), this, SLOT(trt_testProjectManagerGetName()));
	connect(pushbutton_testProjectManagerGetPath, SIGNAL(clicked()), this, SLOT(trt_testProjectManagerGetPath()));
	connect(pushbutton_testSurveyManagerGetName, SIGNAL(clicked()), this, SLOT(trt_testSurveyManagerGetName()));
	connect(pushbutton_testSurveyManagerGetPath, SIGNAL(clicked()), this, SLOT(trt_testSurveyManagerGetPath()));

	connect(pushbutton_testProjectManagerSeismicGetName, SIGNAL(clicked()), this, SLOT(trt_testProjectManagerGetSeismicName()));
	connect(pushbutton_testProjectManagerHorizonGetName, SIGNAL(clicked()), this, SLOT(trt_testProjectManagerGetHorizonName()));
	// connect(pushbutton_testProjectmanagerSeismicGetPath, SIGNAL(clicked()), this, SLOT(trt_testProjectManagerGetSeismicPath()));
	connect(pushbutton_testProjectManagerCulturalGetName, SIGNAL(clicked()), this, SLOT(trt_testProjectManagerGetCulturalName()));
	connect(pushbutton_testProjectManagerWellGet, SIGNAL(clicked()), this, SLOT(trt_testProjectManagerGetWellName()));
	connect(pushbutton_testProjectManagerRgbRawGet, SIGNAL(clicked()), this, SLOT(trt_testProjectManagerGetRGBName()));






	connect(pushbutton_setSeismicWidget, SIGNAL(clicked()), this, SLOT(trt_testCallSeismicWidget()));
	connect(pushbutton_setHorizonWidget, SIGNAL(clicked()), this, SLOT(trt_testCallHorizonWidget()));
	connect(pushbutton_setCulturalsWidget, SIGNAL(clicked()), this, SLOT(trt_testCallCulturalsWidget()));
	connect(pushbutton_setWellWidget, SIGNAL(clicked()), this, SLOT(trt_testCallWellWidget()));
	connect(pushbutton_setRgbSpectrumWidget, SIGNAL(clicked()), this, SLOT(trt_testCallRgbSpectrumWidget()));












	connect(pushbutton_testProjectList, SIGNAL(clicked()), this, SLOT(trt_testProjectList()));
	connect(pushbutton_testSurvey, SIGNAL(clicked()), this, SLOT(trt_testSurvey()));

	connect(pushbutton_projectPath, SIGNAL(clicked()), this, SLOT(trt_testProjectPath()));
	connect(pushbutton_initSurvey, SIGNAL(clicked()), this, SLOT(trt_initSurvey()));
	connect(pushbutton_surveyPath, SIGNAL(clicked()), this, SLOT(trt_testSurveyPath()));

	connect(pushbutton_testSeismic, SIGNAL(clicked()), this, SLOT(trt_testSeismic()));
	connect(pushbutton_initSeismic, SIGNAL(clicked()), this, SLOT(trt_initSeismic()));
	connect(pushbutton_getSeismic, SIGNAL(clicked()), this, SLOT(trt_getSeismic()));
	connect(pushbutton_setSeismicBasket, SIGNAL(clicked()), this, SLOT(trt_setSeismicBasket()));

	connect(pushbutton_testHorizon, SIGNAL(clicked()), this, SLOT(trt_testHorizon()));
	connect(pushbutton_initHorizon, SIGNAL(clicked()), this, SLOT(trt_initHorizon()));
	connect(pushbutton_getHorizon, SIGNAL(clicked()), this, SLOT(trt_getHorizon()));
	connect(pushbutton_setHorizonBasket, SIGNAL(clicked()), this, SLOT(trt_setHorizonBasket()));

	connect(pushbutton_testCulturals, SIGNAL(clicked()), this, SLOT(trt_testCulturals()));
	connect(pushbutton_initCulturals, SIGNAL(clicked()), this, SLOT(trt_initCulturals()));
	connect(pushbutton_getCulturals, SIGNAL(clicked()), this, SLOT(trt_getCulturals()));
	connect(pushbutton_setCulturalsBasket, SIGNAL(clicked()), this, SLOT(trt_setCulturalsBasket()));

	connect(pushbutton_testWells, SIGNAL(clicked()), this, SLOT(trt_testWells()));
	connect(pushbutton_initWells, SIGNAL(clicked()), this, SLOT(trt_initWells()));
	connect(pushbutton_getWells, SIGNAL(clicked()), this, SLOT(trt_getWells()));
	connect(pushbutton_setWellsBasket, SIGNAL(clicked()), this, SLOT(trt_setWellsBasket()));
}


DebugWidget::~DebugWidget()
{


}


void DebugWidget::trt_testProjectManager()
{
	m_projectManagerWidget = new ProjectManagerWidget();
	m_projectManagerWidget->setVisible(true);
}




void DebugWidget::trt_testProjectManagerGetType()
{
	if ( m_projectManagerWidget )
	{
		m_projectType = m_projectManagerWidget->getProjectType();
		qDebug() << m_projectType;
	}
}

void DebugWidget::trt_testProjectManagerGetName()
{
	if ( m_projectManagerWidget )
	{
		m_projectName = m_projectManagerWidget->getProjectName();
		qDebug() << m_projectName;
	}
}

void DebugWidget::trt_testProjectManagerGetPath()
{
	if ( m_projectManagerWidget )
	{
		m_projectPath = m_projectManagerWidget->getProjectPath();
		qDebug() << m_projectPath;
	}
}


void DebugWidget::trt_testSurveyManagerGetName()
{
	if ( m_projectManagerWidget )
	{
		m_surveyName = m_projectManagerWidget->getSurveyName();
		qDebug() << m_surveyName;
	}
}


void DebugWidget::trt_testSurveyManagerGetPath()
{
	if ( m_projectManagerWidget )
	{
		m_surveyPath = m_projectManagerWidget->getSurveyPath();
		qDebug() << m_surveyPath;
	}
}



void DebugWidget::trt_testProjectManagerGetSeismicName()
{

	m_surveyPath = m_projectManagerWidget->getSurveyPath();
	m_surveyName = m_projectManagerWidget->getSurveyName();
	if ( m_projectManagerWidget )
	{
		m_seismicName = m_projectManagerWidget->getSeismicNames();
		for(int n=0; n<m_seismicName.size(); n++)
			qDebug() << m_seismicName[n];
		m_seismicPath = m_projectManagerWidget->getSeismicPath();
		for(int n=0; n<m_seismicPath.size(); n++)
			qDebug() << m_seismicPath[n];
	}
}

void DebugWidget::trt_testProjectManagerGetHorizonName()
{
	m_surveyPath = m_projectManagerWidget->getSurveyPath();
	m_surveyName = m_projectManagerWidget->getSurveyName();
	if ( m_projectManagerWidget )
	{
		m_horizonName = m_projectManagerWidget->getHorizonNames();
		for(int n=0; n<m_horizonName.size(); n++)
			qDebug() << m_horizonName[n];
		m_horizonPath = m_projectManagerWidget->getHorizonPath();
		for(int n=0; n<m_horizonPath.size(); n++)
			qDebug() << m_horizonPath[n];
	}
}


void DebugWidget::trt_testProjectManagerGetCulturalName()
{
	m_projectPath = m_projectManagerWidget->getProjectPath();
	m_projectName = m_projectManagerWidget->getProjectName();
	if ( m_projectManagerWidget )
	{
		m_culturalsCdatName = m_projectManagerWidget->getCulturalsCdatNames();
		m_culturalsCdatPath = m_projectManagerWidget->getCulturalsCdatPath();
		m_culturalsStrdName = m_projectManagerWidget->getCulturalsStrdNames();
		m_culturalsStrdPath = m_projectManagerWidget->getCulturalsStrdPath();
	}
}


void DebugWidget::trt_testProjectManagerGetWellName()
{
	m_projectPath = m_projectManagerWidget->getProjectPath();
	m_projectName = m_projectManagerWidget->getProjectName();
	if ( m_projectManagerWidget )
	{
		WellBasket = m_projectManagerWidget->get_well_list();
	}
}

void DebugWidget::trt_testProjectManagerGetRGBName()
{
	m_surveyPath = m_projectManagerWidget->getSurveyPath();
	m_surveyName = m_projectManagerWidget->getSurveyName();
	if ( m_projectManagerWidget )
	{
		m_rgbrawnames = m_projectManagerWidget->getRgbRawNames();
		m_rgbrawPath = m_projectManagerWidget->getRgbRawPath();
	}
}


void DebugWidget::trt_testProjectManagerGetSeismicPath()
{
	/*
	if ( m_projectManagerWidget )
	{
		m_seismicName = m_projectManagerWidget->getSeismicPath();
		for(int n=0; n<m_seismicPath.size(); n++)
			qDebug() << m_seismicPath[n];
	}
	*/
}







void DebugWidget::trt_testCallSeismicWidget()
{
	m_seismicManager = new SeismicManager();
	m_seismicManager->setVisible(true);
	QString path = m_surveyPath; // "/data/IMA3G/DIR_PROJET/4D_TRAINING_MOBIM_2020/DATA/3D/FP19_4D_MOBIM_dVdnxTmy/";
	QString name = m_surveyName; // "FP19_4D_MOBIM";
	m_seismicManager->setProjectType(2);
	m_seismicManager->setProjectName("a");
	m_seismicManager->setSurveyPath(path, name);
	m_seismicManager->setForceBasket(m_seismicName);
}

void DebugWidget::trt_testCallHorizonWidget()
{
	m_horizonManager = new HorizonManager();
	m_horizonManager->setVisible(true);
	QString path = m_surveyPath; // "/data/IMA3G/DIR_PROJET/4D_TRAINING_MOBIM_2020/DATA/3D/FP19_4D_MOBIM_dVdnxTmy/";
	QString name = m_surveyName; // "FP19_4D_MOBIM";
	m_horizonManager->setProjectType(2);
	m_horizonManager->setProjectName("a");
	m_horizonManager->setSurveyPath(path, name);
	m_horizonManager->setForceBasket(m_horizonName);

}


void DebugWidget::trt_testCallCulturalsWidget()
{
	m_culturalsManager = new CulturalsManager();
	m_culturalsManager->setVisible(true);
	m_culturalsManager->setProjectPath(m_projectPath, m_projectName);
	m_culturalsManager->setForceBasket(m_culturalsCdatName, m_culturalsCdatPath, m_culturalsStrdName, m_culturalsStrdPath);
}

void DebugWidget::trt_testCallWellWidget()
{
	QString path = m_projectPath; // "/data/IMA3G/DIR_PROJET/4D_TRAINING_MOBIM_2020/DATA/3D/FP19_4D_MOBIM_dVdnxTmy/";
	QString name = m_projectName; // "FP19_4D_MOBIM";
	m_wellsManager = new WellsManager();
	m_wellsManager->setVisible(true);
	m_wellsManager->setProjectType(2);
	m_wellsManager->setProjectPath(path, name);
	m_wellsManager->setForceDataBasket(WellBasket);
}

void DebugWidget::trt_testCallRgbSpectrumWidget()
{
	QString path = m_surveyPath; // "/data/IMA3G/DIR_PROJET/4D_TRAINING_MOBIM_2020/DATA/3D/FP19_4D_MOBIM_dVdnxTmy/";
	QString name = m_surveyName; // "FP19_4D_MOBIM"

	m_rgbRawManager = new RgbRawManager();
	m_rgbRawManager->setVisible(true);
	m_rgbRawManager->setSurvey(path, name);
	m_rgbRawManager->setForceBasket(m_rgbrawnames, m_rgbrawPath);
}





























void DebugWidget::trt_testProjectList()
{
	m_projectManager = new ProjectManager();
	m_projectManager->setVisible(true);

}


void DebugWidget::trt_testProjectPath()
{
	QString tiny, path;
	tiny = m_projectManager->getName();
	path = m_projectManager->getPath();

	qDebug() << "[ " << tiny << " ] - " << path;
}


void DebugWidget::trt_testSurvey()
{
	m_surveyManager = new SurveyManager();
	m_surveyManager->setVisible(true);
}

void DebugWidget::trt_initSurvey()
{
	// QString projectPath = "/data/IMA3G/DIR_PROJET/AE_ASAB_4D_TRAINING/";
	// QString projectName = "AE_ASAB_4D_TRAINING";
	QString projectPath = "/data/IMA3G/DIR_PROJET/4D_TRAINING_MOBIM_2020/";
	QString projectName = "4D_TRAINING_MOBIM_2020";
	m_surveyManager->setProjectSelectedPath(2, projectPath, projectName);
}

void DebugWidget::trt_testSurveyPath()
{
	QString tiny, path;
	tiny = m_surveyManager->getName();
	path = m_surveyManager->getPath();
	qDebug() << "[ " << tiny << " ] - " << path;
}
















void DebugWidget::trt_testSeismic()
{
	m_seismicManager = new SeismicManager();
	m_seismicManager->setVisible(true);
}

void DebugWidget::trt_initSeismic()
{
	QString path = m_surveyPath; // "/data/IMA3G/DIR_PROJET/4D_TRAINING_MOBIM_2020/DATA/3D/FP19_4D_MOBIM_dVdnxTmy/";
	QString name = m_surveyName; // "FP19_4D_MOBIM";

	m_seismicManager->setProjectType(2);
	m_seismicManager->setProjectName("a");
	m_seismicManager->setSurveyPath(path, name);
}

void DebugWidget::trt_getSeismic()
{
	std::vector<QString> tiny = m_seismicManager->getNames();
	std::vector<QString> full = m_seismicManager->getPath();

	for (int i=0; i<tiny.size(); i++)
	{
		qDebug() << "[ " << tiny[i] << " ]   " << full[i];
	}
}

void DebugWidget::trt_setSeismicBasket()
{
	std::vector<QString> tiny;
	tiny.push_back("4DBlockQCs_B96_M18_FULL-Mon-Base");
	tiny.push_back("FP19_B96_FULL");
	tiny.push_back("FP19_B96_M11_DIP");
	tiny.push_back("FP19_B96_FULL");
	tiny.push_back("FP19_M18_FULL");
	m_seismicManager->setForceBasket(tiny);
}




void DebugWidget::trt_testHorizon()
{
	m_horizonManager = new HorizonManager();
	m_horizonManager->setVisible(true);
}

void DebugWidget::trt_initHorizon()
{
	QString path = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/";
	QString name = "UMC_small";
	m_horizonManager->setSurveyPath(path, name);
}

void DebugWidget::trt_getHorizon()
{
	std::vector<QString> tiny = m_horizonManager->getNames();
	std::vector<QString> full = m_horizonManager->getPath();

	for (int i=0; i<tiny.size(); i++)
	{
		qDebug() << "[ " << tiny[i] << " ]   " << full[i];
	}
}

void DebugWidget::trt_setHorizonBasket()
{
	std::vector<QString> tiny;
	tiny.push_back("tata");
	tiny.push_back("toto");
	tiny.push_back("as");
	tiny.push_back("base");
	tiny.push_back("debug");
	tiny.push_back("debug1");
	m_horizonManager->setForceBasket(tiny);
}



void DebugWidget::trt_testCulturals()
{
	m_culturalsManager = new CulturalsManager();
	m_culturalsManager->setVisible(true);
}

void DebugWidget::trt_initCulturals()
{
	QString path = "/data/PLI/DIR_PROJET/UMC-NK/";
	QString name = "UMC-NK";
	m_culturalsManager->setProjectPath(path, name);
}

void DebugWidget::trt_getCulturals()
{
	std::vector<QString> cdatTiny = m_culturalsManager->getCdatNames();
	std::vector<QString> cdatFull = m_culturalsManager->getCdatPath();
	std::vector<QString> strdTiny = m_culturalsManager->getStrdNames();
	std::vector<QString> strdFull = m_culturalsManager->getStrdPath();

	for (int n=0; n<cdatTiny.size(); n++)
	{
		qDebug() << "[ " << cdatTiny[n] << " ]  - " << cdatFull[n];
	}
	for (int n=0; n<strdTiny.size(); n++)
	{
		qDebug() << "[ " << strdTiny[n] << " ]  - " << strdFull[n];
	}
}

void DebugWidget::trt_setCulturalsBasket()
{

}



void DebugWidget::trt_testWells()
{
	m_wellsManager = new WellsManager();
	m_wellsManager->setVisible(true);
}


void DebugWidget::trt_initWells()
{
	QString path = "/data/IMA3G/DIR_PROJET/4D_TRAINING_MOBIM_2020/";
	QString name = "4D_TRAINING_MOBIM_2020";
	m_wellsManager->setProjectType(2);
	m_wellsManager->setProjectPath(path, name);
}

void DebugWidget::trt_getWells()
{
	WellBasket = m_wellsManager->getBasket();
}

void DebugWidget::trt_setWellsBasket()
{
	// WellBasket.clear();
	m_wellsManager->setForceDataBasket(WellBasket);
}

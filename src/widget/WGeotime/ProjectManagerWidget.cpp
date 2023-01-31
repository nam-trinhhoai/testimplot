

#include <QSettings>
#include <QProcess>
#include <QFileDialog>

#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>

#include <QDebug>

#include <ProjectManagerWidget.h>
#include <spectrumHorizonUtil.h>
#include <geotimepath.h>
#include "globalconfig.h"

ProjectManagerWidget::ProjectManagerWidget(bool sessionVisible, QWidget* parent) :
		QWidget(parent) {
	// init maps
	m_nameToIndex[ManagerTabName::SEISMIC] = 0;
	m_nameToIndex[ManagerTabName::HORIZON] = 1;
	m_nameToIndex[ManagerTabName::CULTURAL] = 2;
	m_nameToIndex[ManagerTabName::WELL] = 3;
	m_nameToIndex[ManagerTabName::PICK] = 4;
	m_nameToIndex[ManagerTabName::RGBRAW] = 5;

	m_isPageShownMap[ManagerTabName::SEISMIC] = false;
	m_isPageShownMap[ManagerTabName::HORIZON] = false;
	m_isPageShownMap[ManagerTabName::CULTURAL] = false;
	m_isPageShownMap[ManagerTabName::WELL] = false;
	m_isPageShownMap[ManagerTabName::PICK] = false;
	m_isPageShownMap[ManagerTabName::RGBRAW] = false;

	QVBoxLayout *mainLayout00 = new QVBoxLayout(this);

	QPushButton *pushbutton_loadsession = new QPushButton("load session");

	QHBoxLayout *mainLayout01 = new QHBoxLayout;
	m_projectManager = new ProjectManager();
	m_surveyManager = new SurveyManager();
	m_projectManager->setSurveyManager(m_surveyManager);

	m_qgb_seismic = new QGroupBox();
	QVBoxLayout *seismicLayout = new QVBoxLayout(m_qgb_seismic);
	m_seismicManager = new SeismicManager();
	m_surveyManager->setSeismicManager(m_seismicManager);
	seismicLayout->addWidget(m_seismicManager);

	m_qgb_horizon = new QGroupBox();
	QVBoxLayout *horizonLayout = new QVBoxLayout(m_qgb_horizon);
	m_horizonManager = new HorizonManager();
	m_surveyManager->setHorizonManager(m_horizonManager);
	horizonLayout->addWidget(m_horizonManager);

	m_qgb_culturals = new QGroupBox();
	QVBoxLayout *culturalsLayout = new QVBoxLayout(m_qgb_culturals);
	m_culturalsManager = new CulturalsManager();
	m_projectManager->setCulturalsManager(m_culturalsManager);
	culturalsLayout->addWidget(m_culturalsManager);

	m_qgb_wells = new QGroupBox();
	QVBoxLayout *wellsLayout = new QVBoxLayout(m_qgb_wells);
	m_wellsManager = new WellsManager();
	m_projectManager->setWellsManager(m_wellsManager);
	wellsLayout->addWidget(m_wellsManager);


	m_qgb_picks = new QGroupBox();
	QVBoxLayout *picksLayout = new QVBoxLayout(m_qgb_picks);
	m_picksManager = new PicksManager();
	m_projectManager->setPicksManager(m_picksManager);
	picksLayout->addWidget(m_picksManager);

	m_qgb_rgbRaw = new QGroupBox();
	QVBoxLayout *RgbRawLayout = new QVBoxLayout(m_qgb_rgbRaw);
	m_rgbRawManager = new RgbRawManager();
	m_surveyManager->setRgbRawManager(m_rgbRawManager);
	RgbRawLayout->addWidget(m_rgbRawManager);

	tabwidget = new QTabWidget();
	/*
	showTab(ManagerTabName::SEISMIC);
	showTab(ManagerTabName::HORIZON);
	showTab(ManagerTabName::CULTURAL);
	showTab(ManagerTabName::WELL);
	showTab(ManagerTabName::PICK);
	showTab(ManagerTabName::RGBRAW);
	*/

	QPushButton *pushbutton_savesession = new QPushButton("save session");
	QPushButton *pushbutton_debug = new QPushButton("debug");

	mainLayout01->addWidget(m_projectManager);
	mainLayout01->addWidget(m_surveyManager);
	mainLayout00->addWidget(pushbutton_loadsession);
	mainLayout00->addLayout(mainLayout01);
	// mainLayout00->addWidget(tabwidget);
	mainLayout00->addWidget(pushbutton_savesession);
	// mainLayout00->addWidget(pushbutton_debug);

	pushbutton_loadsession->setVisible(sessionVisible);
	pushbutton_savesession->setVisible(sessionVisible);


	// m_projectManager->setCallBackListClick(trt_projetlistClick);
    // connect(m_projectManager->listwidgetProject, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_projetlistClick(QListWidgetItem*)));
	connect(pushbutton_loadsession, SIGNAL(clicked()), this, SLOT(trt_session_load()));
	connect(pushbutton_savesession, SIGNAL(clicked()), this, SLOT(trt_session_save()));
	connect(pushbutton_debug, SIGNAL(clicked()), this, SLOT(trt_debug()));

	// auto load last session
	load_last_session();
	// connect(pushbutton_debug, SIGNAL(clicked()), this, SLOT(trt_debug()));
}




ProjectManagerWidget::~ProjectManagerWidget()
{
	if (m_qgb_seismic->parent()==nullptr) {
		m_qgb_seismic->deleteLater();
	}
	if (m_qgb_horizon->parent()==nullptr) {
		m_qgb_horizon->deleteLater();
	}
	if (m_qgb_culturals->parent()==nullptr) {
		m_qgb_culturals->deleteLater();
	}
	if (m_qgb_wells->parent()==nullptr) {
		m_qgb_wells->deleteLater();
	}
	if (m_qgb_rgbRaw->parent()==nullptr) {
		m_qgb_rgbRaw->deleteLater();
	}
}

void ProjectManagerWidget::onlyShow(const std::vector<ProjectManagerWidget::ManagerTabName>& tabNames) {
	std::map<ManagerTabName, bool> notDefinedMap;
	std::map<ManagerTabName, int>::const_iterator it = m_nameToIndex.begin();
	while (it!=m_nameToIndex.end()) {
		notDefinedMap[it->first] = true;
		it++;
	}

	for (int i=0; i<tabNames.size(); i++) {
		if (notDefinedMap[tabNames[i]]) {
			notDefinedMap[tabNames[i]] = false;
			showTab(tabNames[i]);
		}
	}

	std::map<ManagerTabName, bool>::const_iterator itBool = notDefinedMap.begin();
	while (itBool!=notDefinedMap.end()) {
		if (itBool->second) {
			hideTab(itBool->first);
		}
		itBool++;
	}
}

QWidget* ProjectManagerWidget::widgetFromTabName(ProjectManagerWidget::ManagerTabName tabName) {
	switch (tabName) {
	case ManagerTabName::SEISMIC:
		return m_qgb_seismic;
		break;
	case ManagerTabName::HORIZON:
		return m_qgb_horizon;
		break;
	case ManagerTabName::CULTURAL:
		return m_qgb_culturals;
		break;
	case ManagerTabName::WELL:
		return m_qgb_wells;
		break;
	case ManagerTabName::PICK:
		return m_qgb_picks;
		break;
	case ManagerTabName::RGBRAW:
		return m_qgb_rgbRaw;
		break;
	}
}

void ProjectManagerWidget::showTab(ProjectManagerWidget::ManagerTabName tabName) {
	QString name;
	int realIndex = m_nameToIndex[tabName];
	QWidget* widget = widgetFromTabName(tabName);
	switch (tabName) {
	case ManagerTabName::SEISMIC:
		name = "seismic";
		break;
	case ManagerTabName::HORIZON:
		name = "horizon";
		break;
	case ManagerTabName::CULTURAL:
		name = "culturals";
		break;
	case ManagerTabName::WELL:
		name = "wells";
		break;
	case ManagerTabName::PICK:
		name = "picks";
		break;
	case ManagerTabName::RGBRAW:
		name = "Rgb Spectrum";
		break;
	}
	if (!m_isPageShownMap[tabName]) {
		int index = 0;
		std::map<ManagerTabName, int>::const_iterator it = m_nameToIndex.begin();
		while (it!=m_nameToIndex.end()) {
			if (m_isPageShownMap[it->first] && it->second<realIndex) {
				index++;
			}
			it++;
		}
		tabwidget->insertTab(index, widget, QIcon(QString("")), name);
		m_isPageShownMap[tabName] = true;
	}
}

void ProjectManagerWidget::hideTab(ProjectManagerWidget::ManagerTabName tabName) {
	int removeIndex = tabwidget->indexOf(widgetFromTabName(tabName));
	if (m_isPageShownMap[tabName] && removeIndex>=0) {
		tabwidget->removeTab(removeIndex);
		m_isPageShownMap[tabName] = false;
	}
}


QString ProjectManagerWidget::getNextVisionPath()
{
	return getSurveyPath() + "/" + QString::fromStdString(GeotimePath::NEXTVISION_IMPORT_EXPORT_DIR) + "/" + QString::fromStdString(GeotimePath::NEXTVISION_MAIN_DIR) + "/";
}

QString ProjectManagerWidget::getNVHorizonPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_DIR) + "/";
}

QString ProjectManagerWidget::getIsoHorizonPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_ISOHORIZON_DIR) + "/";
}

QString ProjectManagerWidget::getNextVisionSeismicPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_SEISMIC_DIR) + "/";
}

QString ProjectManagerWidget::getPatchPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_PATCH_DIR) + "/";
}

QString ProjectManagerWidget::getVideoPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_VIDEO_DIR) + "/";
}


bool ProjectManagerWidget::mkdirNextVisionPath()
{
	QDir d;
	return d.mkpath(getNextVisionPath());
}

bool ProjectManagerWidget::mkdirNVHorizonPath()
{
	QDir d;
	return d.mkpath(getNVHorizonPath());
}

bool ProjectManagerWidget::mkdirIsoHorizonPath()
{
	QDir d;
	return d.mkpath(getIsoHorizonPath());
}

bool ProjectManagerWidget::mkdirNextVisionSeismicPath()
{
	QDir d;
	return d.mkpath(getNextVisionSeismicPath());
}

bool ProjectManagerWidget::mkdirPatchPath()
{
	QDir d;
	return d.mkpath(getPatchPath());
}

bool ProjectManagerWidget::mkdirVideoPath()
{
	QDir d;
	return d.mkpath(getVideoPath());
}


// compatibility version 0
std::vector<QString> ProjectManagerWidget::get_seismic_names()
{
	return getSeismicNames();
}

std::vector<QString> ProjectManagerWidget::get_seismic_fullpath_names()
{
	std::vector<QString> a = getSeismicPath();
	for (int i=0; i<a.size(); i++)
	{
		qDebug() << a[i];
	}


	return getSeismicPath();
}


QString ProjectManagerWidget::horizon_path_read()
{
	return "";
}

std::vector<QString> ProjectManagerWidget::get_horizon_names()
{
	return getHorizonNames();
}

std::vector<QString> ProjectManagerWidget::get_horizon_fullpath_names()
{
	return getHorizonPath();
}

QString ProjectManagerWidget::getProjectType()
{
	if ( m_projectManager )
		return m_projectManager->getType();
	else
		return "";
}


QString ProjectManagerWidget::getProjectName()
{
	if ( m_projectManager )
		return m_projectManager->getName();
	else
		return "";
}


QString ProjectManagerWidget::getProjectPath()
{
	if ( m_projectManager )
		return m_projectManager->getPath();
	else
		return "";
}


QString ProjectManagerWidget::getSurveyName()
{
	if ( m_surveyManager )
		return m_surveyManager->getName();
	else
		return "";
}


QString ProjectManagerWidget::getSurveyPath()
{
	if ( m_surveyManager )
		return m_surveyManager->getPath();
	else
		return "";
}

QString ProjectManagerWidget::getSeismicDirectory()
{
	if ( m_seismicManager )
	return m_seismicManager->getSeismicDirectory();
	else return "";
}

QString ProjectManagerWidget::getImportExportPath()
{
	QString surveyPath = getSurveyPath();
	if ( surveyPath.isEmpty() ) return "";
	return surveyPath + "ImportExport/";
}

QString ProjectManagerWidget::getIJKPath()
{
	QString path = getImportExportPath();
	if ( path.isEmpty() ) return "";
	return path + "IJK/";
}

/*
QString ProjectManagerWidget::getPatchPath()
{
	QString path = getImportExportPath();
	if ( path.isEmpty() ) return "";
	return path + "Patch/";
}
*/

QString ProjectManagerWidget::getHorizonsPath()
{
	/*
	QString path = getIJKPath();
	if ( path.isEmpty() ) return "";
	return path + SPECTRUM_HORIZON_MAIN_DIR + "/";
	*/
	return getSurveyPath() +"/" + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_PATH) + "/";
}

QString ProjectManagerWidget::getHorizonsIsoValPath()
{
	/*
	QString path = getHorizonsPath();
	if ( path.isEmpty() ) return "";
	return path + SPECTRUM_HORIZON_ISOVAL_DIR + "/";
	*/
	return getSurveyPath() +"/" + QString::fromStdString(GeotimePath::NEXTVISION_ISOHORIZON_PATH) + "/";
}


QString ProjectManagerWidget::getHorizonsFreePath()
{
	QString path = getHorizonsPath();
	if ( path.isEmpty() ) return "";
	return path + SPECTRUM_HORIZON_FREE_DIR + "/";
}

std::vector<QString> ProjectManagerWidget::getFreeHorizonNames()
{
	if ( m_horizonManager ) return m_horizonManager->getFreeName();
	std::vector<QString> ret;
	return ret;
}

std::vector<QString> ProjectManagerWidget::getFreeHorizonFullName()
{
	if ( m_horizonManager ) return m_horizonManager->getFreePath();
	std::vector<QString> ret;
	return ret;
}


void ProjectManagerWidget::createPatchDir()
{
	QString path = getPatchPath();
	if ( path.isEmpty() ) return;
	QDir dir(path);
	if ( !dir.exists() ) dir.mkpath(".");
}

std::vector<QString> ProjectManagerWidget::getSeismicNames()
{
	std::vector<QString> ret;
	if ( m_seismicManager )
		ret = m_seismicManager->getNames();
	return ret;
}


std::vector<QString> ProjectManagerWidget::getSeismicPath()
{
	std::vector<QString> ret;
	if ( m_seismicManager )
		ret = m_seismicManager->getPath();
	return ret;
}

std::vector<QString> ProjectManagerWidget::getSeismicAllNames()
{
	std::vector<QString> ret;
	if ( m_seismicManager )
		ret = m_seismicManager->getAllNames();
	return ret;
}


std::vector<QString> ProjectManagerWidget::getSeismicAllPath()
{
	std::vector<QString> ret;
	if ( m_seismicManager )
		ret = m_seismicManager->getAllPath();
	return ret;
}

std::vector<int> ProjectManagerWidget::getSeismicAllDimx()
{
	std::vector<int> ret;
	if ( m_seismicManager )
		ret = m_seismicManager->getAllDimx();
	return ret;
}

std::vector<int> ProjectManagerWidget::getSeismicAllDimy()
{
	std::vector<int> ret;
	if ( m_seismicManager )
		ret = m_seismicManager->getAllDimy();
	return ret;
}

std::vector<int> ProjectManagerWidget::getSeismicAllDimz()
{
	std::vector<int> ret;
	if ( m_seismicManager )
		ret = m_seismicManager->getAllDimz();
	return ret;
}

std::vector<QString> ProjectManagerWidget::getHorizonNames()
{
	std::vector<QString> ret;
	if ( m_horizonManager )
		ret = m_horizonManager->getNames();
	return ret;
}


std::vector<QString> ProjectManagerWidget::getHorizonPath()
{
	std::vector<QString> ret;
	if ( m_horizonManager )
		ret = m_horizonManager->getPath();
	return ret;
}


std::vector<QString> ProjectManagerWidget::getHorizonAllNames()
{
	std::vector<QString> ret;
	if ( m_horizonManager )
		ret = m_horizonManager->getAllNames();
	return ret;
}

std::vector<QString> ProjectManagerWidget::getHorizonAllPath()
{
	std::vector<QString> ret;
	if ( m_horizonManager )
		ret = m_horizonManager->getAllPath();
	return ret;
}

std::vector<QString> ProjectManagerWidget::getHorizonIsoValueListName()
{
	std::vector<QString> ret;
	if ( m_horizonManager )
		ret = m_horizonManager->getIsoValueListName();
	return ret;
}

std::vector<QString> ProjectManagerWidget::getHorizonIsoValueListPath()
{
	std::vector<QString> ret;
	if ( m_horizonManager )
		ret = m_horizonManager->getIsoValueListPath();
	return ret;
}




std::vector<QString> ProjectManagerWidget::getCulturalsCdatNames()
{
	std::vector<QString> ret;
	if ( m_culturalsManager )
	{
		return m_culturalsManager->getCdatNames();
	}

}

std::vector<QString> ProjectManagerWidget::getCulturalsCdatPath()
{
	std::vector<QString> ret;
	if ( m_culturalsManager )
	{
		return m_culturalsManager->getCdatPath();
	}
}

std::vector<QString> ProjectManagerWidget::getCulturalsStrdNames()
{
	std::vector<QString> ret;
	if ( m_culturalsManager )
	{
		return m_culturalsManager->getStrdNames();
	}
}

std::vector<QString> ProjectManagerWidget::getCulturalsStrdPath()
{
	std::vector<QString> ret;
	if ( m_culturalsManager )
	{
		return m_culturalsManager->getStrdPath();
	}
}


WELLMASTER ProjectManagerWidget::get_well_list()
{
	WELLMASTER ret;
	if ( m_wellsManager )
		ret = m_wellsManager->getBasket();
	return ret;
}

std::vector<QString> ProjectManagerWidget::getRgbRawNames()
{
	std::vector<QString> ret;
	if ( m_rgbRawManager )
	{
		ret = m_rgbRawManager->getNames();
	}
	return ret;
}

std::vector<QString> ProjectManagerWidget::getRgbRawPath()
{
	std::vector<QString> ret;
	if ( m_rgbRawManager )
	{
		ret = m_rgbRawManager->getPath();
	}
	return ret;
}

std::vector<QString> ProjectManagerWidget::getAviNames()
{
	std::vector<QString> ret;
	if ( m_rgbRawManager )
	{
		ret = m_rgbRawManager->getAviNames();
	}
	return ret;
}

std::vector<QString> ProjectManagerWidget::getAviPath()
{
	std::vector<QString> ret;
	if ( m_rgbRawManager )
	{
		ret = m_rgbRawManager->getAviPath();
	}
	return ret;
}

std::vector<QString> ProjectManagerWidget::getRgbRawDirectoryNames()
{
	std::vector<QString> ret;
	if ( m_rgbRawManager )
	{
		ret = m_rgbRawManager->getAllDirectoryNames();
	}
	return ret;
}

std::vector<QString> ProjectManagerWidget::getRgbRawDirectoryPath()
{
	std::vector<QString> ret;
	if ( m_rgbRawManager )
	{
		ret = m_rgbRawManager->getAllDirectoryPath();
	}
	return ret;
}


void ProjectManagerWidget::RgbRawUpdateNames()
{
	if ( m_rgbRawManager )
	{
		m_rgbRawManager->updateNames();
	}
}

std::vector<WELLHEADDATA> ProjectManagerWidget::getMainWellData()
{
	if ( m_wellsManager) return m_wellsManager->getMainData();
	std::vector<WELLHEADDATA> ret;
	return ret;
}

std::vector<MARKER> ProjectManagerWidget::getPicksSortedWells()
{
	std::vector<QString> picksNames = getPicksBasketNames();
	const std::vector<QBrush>& colors = getPicksBasketColors();
	return m_wellsManager->getPicksSortedWells(picksNames, colors);
}


std::vector<std::vector<QString>> ProjectManagerWidget::getWellBasketLogPicksTf2pNames(QString type, QString nameType)
{
	if ( m_wellsManager ) return m_wellsManager->getWellBasketLogPicksTf2pNames(type, nameType);
	std::vector<std::vector<QString>> out;
	return out;
}

std::vector<QString> ProjectManagerWidget::getWellBasketTinyNames()
{
	if ( m_wellsManager ) return m_wellsManager->getWellBasketTinyNames();
	std::vector<QString> out;
	return out;
}

std::vector<QString> ProjectManagerWidget::getWellBasketFullNames()
{

	if ( m_wellsManager ) return m_wellsManager->getWellBasketFullNames();
	std::vector<QString> out;
	return out;
}



std::vector<QString> ProjectManagerWidget::getPicksBasketFullNames()
{

	if ( m_picksManager ) return m_picksManager->getPath();
	std::vector<QString> out;
	return out;
}

std::vector<QString> ProjectManagerWidget::getPicksBasketNames()
{

	if ( m_picksManager ) return m_picksManager->getNames();
	std::vector<QString> out;
	return out;
}

std::vector<QBrush> ProjectManagerWidget::getPicksBasketColors()
{

	if ( m_picksManager ) return m_picksManager->getColors();
	std::vector<QBrush> out;
	return out;
}


// todo
void ProjectManagerWidget::fill_empty_logs_list()
{

}


void ProjectManagerWidget::trt_debug()
{
	std::vector<MARKER> data = getPicksSortedWells();

	char *filename = "/data/PLI/NKDEEP/jacques/wells0.txt";
	FILE *pf = fopen(filename, "w");
	int N = data.size();
	for (int n=0; n<N; n++)
	{
		for (int ii=0; ii<10; ii++)
			fprintf(pf, "=================== PICKS ==================\n");

		int N2 = data[n].wellPickLists.size();
		for (int n2=0; n2<N2; n2++)
		{
			fprintf(pf, "well: %d\n%s\n%s\n", n2, data[n].wellPickLists[n2].name.toStdString().c_str(), data[n].wellPickLists[n2].path.toStdString().c_str());
			std::vector<WELLBOREPICKSLIST> wellBore = data[n].wellPickLists[n2].wellBore;
			for (int i=0; i<wellBore.size(); i++)
			{
				fprintf(pf, "[%d]\n", i);
				fprintf(pf, "%s\n", wellBore[i].boreName.toStdString().c_str());
				fprintf(pf, "%s\n", wellBore[i].borePath.toStdString().c_str());
				fprintf(pf, "%s\n", wellBore[i].deviationPath.toStdString().c_str());
				fprintf(pf, "%s\n", wellBore[i].picksName.toStdString().c_str());
				fprintf(pf, "%s\n", wellBore[i].picksPath.toStdString().c_str());
				fprintf(pf, "=========================================\n");
			}
			fprintf(pf, "\n\n");
		}
	}
	fclose(pf);

	/*
	std::vector<QString> data = getPicksBasketNames();
	for (QString st:data)
	{
		qDebug() << st;
	}
	*/

	/*
	char *filename = "/data/PLI/NKDEEP/jacques/wells0.txt";
	FILE *pf = fopen(filename, "w");
	std::vector<WELLHEADDATA> data0 = getMainWellData();
	for (int  n=0; n<data0.size(); n++)
	{
		qDebug() << QString::number(n) + " " + data0[n].tinyName + " -->   " + data0[n].fullName;
		fprintf(pf, "well: %d : %s   --->  %s\n", n, data0[n].tinyName.toStdString().c_str(), data0[n].fullName.toStdString().c_str());
		std::vector<WELLBOREDATA> bore = data0[n].bore;

		for (int j=0; j<bore.size(); j++)
		{
			QString deviation = bore[j].fullName + "/deviation";
			fprintf(pf, "bore: %d : %s   --->  %s [deviation: %s %s]\n", j, bore[j].tinyName.toStdString().c_str(), bore[j].fullName.toStdString().c_str(), deviation, "deviation");
			std::vector<QString> logsTiny = bore[j].logs.getTiny();
			std::vector<QString> logsFull = bore[j].logs.getFull();
			for (int j1=0; j1<logsTiny.size(); j1++)
			{
				fprintf(pf, "log: %d %s %s\n", j1, logsTiny[j1].toStdString().c_str(), logsFull[j1].toStdString().c_str());
			}
			std::vector<QString> tf2pTiny = bore[j].tf2p.getTiny();
			std::vector<QString> tf2pFull = bore[j].tf2p.getFull();
			for (int j1=0; j1<tf2pTiny.size(); j1++)
			{
				fprintf(pf, "tf2p: %d %s %s\n", j1, tf2pTiny[j1].toStdString().c_str(), tf2pFull[j1].toStdString().c_str());
			}
			std::vector<QString> picksTiny = bore[j].picks.getTiny();
			std::vector<QString> picksFull = bore[j].picks.getFull();
			for (int j1=0; j1<picksTiny.size(); j1++)
			{
				fprintf(pf, "picks: %d %s %s\n", j1, picksTiny[j1].toStdString().c_str(), picksFull[j1].toStdString().c_str());
			}
		}
	}

	fclose(pf);


	/*
	std::vector<std::vector<QString>> data = getWellBasketLogPicksTf2pNames("log", "full");
	std::vector<QString> wellBasketFull = getWellBasketFullNames();
	std::vector<QString> wellBasketTiny = getWellBasketTinyNames();

	for (int i=0; i<wellBasketFull.size(); i++)
	{
		qDebug() << wellBasketTiny[i] + "[ " + wellBasketFull[i] + " ]";
	}


	for (int i=0; i<data.size(); i++)
	{
		for (int j=0; j<data[i].size(); j++)
		{
			qDebug() << QString::number(i) + " - " + QString::number(j) + "  -> " + data[i][j];
		}
	}
	*/
}
























void ProjectManagerWidget::trt_projetlistClick(QListWidgetItem*item)
{

}



void ProjectManagerWidget::trt_session_load()
{
    load_session_gui();
}

void ProjectManagerWidget::load_session_gui()
{
    QSettings settings;
    GlobalConfig& config = GlobalConfig::getConfig();
    const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, config.sessionPath()).toString();

    const QString filePath = QFileDialog::getOpenFileName(this,
                                                          tr("Load Session"),
                                                          lastPath,
                                                          QLatin1String("*.json"));
    if (filePath.isEmpty())
        return;

    const QFileInfo fi(filePath);
    settings.setValue(LAST_SESSION_PATH_IN_SETTINGS, fi.absoluteFilePath());

    load_session(filePath);
}

void ProjectManagerWidget::load_last_session() {
	QSettings settings;
	const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, "").toString();
	if (lastPath.isEmpty() || !QFileInfo(lastPath).exists())
		return;

	load_session(lastPath);
}

void ProjectManagerWidget::trt_session_save()
{
	save_session_gui();
}

void ProjectManagerWidget::save_session_gui()
{
	QSettings settings;
	GlobalConfig& config = GlobalConfig::getConfig();
	QString defaultSessionPath = config.sessionPath();
	if (!QFileInfo(defaultSessionPath).exists()) {
		QDir dir = QFileInfo(defaultSessionPath).absoluteDir();
		dir.mkdir(QFileInfo(defaultSessionPath).fileName());
	}

	const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, defaultSessionPath).toString();

	QFileDialog fileDialog(this, tr("Save Session"), lastPath, QLatin1String("*.json"));
	fileDialog.setDefaultSuffix("json");
	fileDialog.setAcceptMode(QFileDialog::AcceptSave);
	int result = fileDialog.exec();

	QString filePath;
	if (result==QDialog::Accepted && fileDialog.selectedFiles().size()>0) {
		filePath = fileDialog.selectedFiles()[0];
	} else {
		return;
	}

	const QFileInfo fi(filePath);
	settings.setValue(LAST_SESSION_PATH_IN_SETTINGS, fi.absoluteFilePath());

	save_session(filePath);
}


void ProjectManagerWidget::save_session(QString sessionPath) {
	QFile file(sessionPath);
	if (!file.open(QIODevice::WriteOnly)) {
		qDebug() << "GeotimeProjectManagerWidget : cannot save session, file not writable";
		return;
	}

	QJsonObject obj;
	obj.insert(projectTypeKey0, getProjectType());
	if ( getProjectType().compare("USER") == 0 )
	{
		obj.insert(projectPathKey0, getProjectPath());
	}

	if (getProjectType().compare("None")!=0 && getProjectName() != nullptr)
	{
		obj.insert(projectKey0, getProjectName());

		if (getSurveyName().compare("") != 0 )
		{
			obj.insert(surveyKey0, getSurveyName());
			QJsonArray seismics, horizons, culturals, wells, neurons, picksNames, picksPath;

			std::vector<QString> names = getSeismicNames();
			for (std::size_t arrayIdx=0; arrayIdx<names.size(); arrayIdx++) {
				seismics.append(names[arrayIdx]);
			}
			names = getHorizonNames();
			for (std::size_t arrayIdx=0; arrayIdx<names.size(); arrayIdx++) {
				horizons.append(names[arrayIdx]);
			}
			names = getCulturalsCdatNames();
			for (std::size_t arrayIdx=0; arrayIdx<names.size(); arrayIdx++) {
					culturals.append(names[arrayIdx]);
			}
			names = getCulturalsStrdNames();
			for (std::size_t arrayIdx=0; arrayIdx<names.size(); arrayIdx++) {
					culturals.append(names[arrayIdx]);
			}
			// picks new tab
			std::vector<QString> _picksNames = m_picksManager->getNames();
			for (std::size_t arrayIdx=0; arrayIdx<_picksNames.size(); arrayIdx++) {
				picksNames.append(_picksNames[arrayIdx]);
			}
			std::vector<QString> _picksPath = m_picksManager->getPath();
			for (std::size_t arrayIdx=0; arrayIdx<_picksPath.size(); arrayIdx++) {
				picksPath.append(_picksPath[arrayIdx]);
			}


			WELLMASTER wellMaster = get_well_list();

			std::vector<WELLLIST> wellList = wellMaster.finalWellBasket;
			std::vector<QString> boreFullName = wellMaster.m_basketWellBore.getFull();

			for (std::size_t arrayIdx=0; arrayIdx<boreFullName.size(); arrayIdx++)
			{
				// find well head and well bore indexes
				int idx_well = 0;
				int idx_bore = 0;
				bool found = WellUtil::getIndexFromWellLists(wellList, boreFullName[arrayIdx], &idx_well, &idx_bore);

				if (!found) {
					continue;
				}

				QString wellBoreUIName = wellList[idx_well].wellborelist[idx_bore].bore_tinyname;
				QString wellBorePath = wellList[idx_well].wellborelist[idx_bore].bore_fullname;

				QJsonObject wellObj;
				wellObj.insert(wellKey0, wellBoreUIName);
				wellObj.insert(wellPathKey0, wellBorePath);

				QJsonArray logsArray, tfpArray, picksArray;

				const std::vector<QString>& log_tinyname = wellList[idx_well].wellborelist[idx_bore].log_tinyname;
				const std::vector<QString>& log_fullname = wellList[idx_well].wellborelist[idx_bore].log_fullname;
				for (long i=0; i<log_tinyname.size(); i++) {
					QJsonObject logObj;
					logObj.insert(tinynameKey0, log_tinyname[i]);
					logObj.insert(fullnameKey0, log_fullname[i]);
					logsArray.append(logObj);
				}
				wellObj.insert(wellLogKey0, logsArray);

				const std::vector<QString>& tfp_tinyname = wellList[idx_well].wellborelist[idx_bore].tf2p_tinyname;
				const std::vector<QString>& tfp_fullname = wellList[idx_well].wellborelist[idx_bore].tf2p_fullname;
				for (long i=0; i<tfp_tinyname.size(); i++) {
					QJsonObject tfpObj;
					tfpObj.insert(tinynameKey0, tfp_tinyname[i]);
					tfpObj.insert(fullnameKey0, tfp_fullname[i]);
					tfpArray.append(tfpObj);
				}
				wellObj.insert(wellTFPKey0, tfpArray);

//				const std::vector<QString>& pick_tinyname = wellList[idx_well].wellborelist[idx_bore].picks_tinyname;
//				const std::vector<QString>& pick_fullname = wellList[idx_well].wellborelist[idx_bore].picks_fullname;
//				for (long i=0; i<pick_tinyname.size(); i++) {
//					QJsonObject pickObj;
//					pickObj.insert(tinynameKey0, pick_tinyname[i]);
//					pickObj.insert(fullnameKey0, pick_fullname[i]);
//					picksArray.append(pickObj);
//				}
//				wellObj.insert(wellPickKey0, picksArray);
				wells.append(wellObj);
			}

			/*
			for (std::size_t arrayIdx=0; arrayIdx<lw_neurons->count(); arrayIdx++) {
				QListWidgetItem* neuronItem = lw_neurons->item(arrayIdx);
				if (neuronItem->isSelected()) {
					neurons.append(neuronItem->text());
				}
			}
			*/
			obj.insert(seismicKey0, seismics);
			obj.insert(horizonKey0, horizons);
			obj.insert(culturalKey0, culturals);
			obj.insert(wellKey0, wells);
			// obj.insert(neuronKey, neurons);
			obj.insert(picksNamesKey0, picksNames);
			obj.insert(picksPathKey0, picksPath);
		}
	}
    QJsonDocument doc(obj);
    file.write(doc.toJson());
}





void ProjectManagerWidget::load_session(QString sessionPath)
{
	QFile file(sessionPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        qDebug() << "GeotimeProjectManagerWidget : cannot load session, file not readable";
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isObject())
    {
        qDebug() << "GeotimeProjectManagerWidget : cannot load session, root is not a json object";
        return;
    }

    bool isProjectValid = false;
    bool isSurveyValid = false;

    QJsonObject rootObj = doc.object();
    QString project = "";
    QString survey = "";

    if (rootObj.contains(projectTypeKey0))
    {
    	int idx = 0;
    	QString projectPath = "";
    	QString project_type = rootObj.value(projectTypeKey0).toString("None");
    	if ( project_type.compare("USER") == 0 && rootObj.contains(projectPathKey0) )
    	{
    		projectPath = rootObj.value(projectPathKey0).toString(".");
    		m_projectManager->setUserProjectPath(projectPath);
    	}
    	while(idx<m_projectManager->comboboxProjectType->count() && project_type.compare(m_projectManager->comboboxProjectType->itemText(idx))!=0)
    	{
    		idx ++;
    	}
    	if (idx>=m_projectManager->comboboxProjectType->count())
    	{
    		idx = 0; // set as None
    	}
    	m_projectManager->comboboxProjectType->setCurrentIndex(idx);

    	if (idx!=0 && rootObj.contains(projectKey0))
    	{ // load project
   	   		QString project = rootObj.value(projectKey0).toString("");
    		std::size_t projectIdx = 0;
    		while(projectIdx<m_projectManager->listwidgetProject->count() && project.compare(m_projectManager->listwidgetProject->item(projectIdx)->text())!=0)
    		{
   	    		projectIdx++;
 	    	}
    		if (projectIdx<m_projectManager->listwidgetProject->count())
    		{
    	    //		disconnect(lineedit_surveysearch, SIGNAL(textChanged(QString)), this, SLOT(trt_surveysearchchange(QString)));
    	    //		lineedit_surveysearch->setText("");
    	    //		connect(lineedit_surveysearch, SIGNAL(textChanged(QString)), this, SLOT(trt_surveysearchchange(QString)));
    	    			//lw_projetlist->item(projectIdx)->setSelected(true);
    			m_projectManager->listwidgetProject->setCurrentItem(m_projectManager->listwidgetProject->item(projectIdx));
    			m_projectManager->trt_projetlistClick(m_projectManager->listwidgetProject->item(projectIdx));
    			isProjectValid = true;
    		}
    	}
    	if (isProjectValid && rootObj.contains(surveyKey0))
    	{
    		QString survey = rootObj.value(surveyKey0).toString("");
    		std::size_t surveyIdx = 0;
    		while(surveyIdx<m_surveyManager->listwidgetSurvey->count() && survey.compare(m_surveyManager->listwidgetSurvey->item(surveyIdx)->text())!=0)
    		{
    			surveyIdx++;
    		}
    		if (surveyIdx<m_surveyManager->listwidgetSurvey->count())
    		{
    			m_surveyManager->listwidgetSurvey->setCurrentItem(m_surveyManager->listwidgetSurvey->item(surveyIdx));
    			m_surveyManager->trt_surveylistClick(m_surveyManager->listwidgetSurvey->item(surveyIdx));
    			// this->qlw_wellpicks->clearSelection();
    			isSurveyValid = true;
    		}
    	}


    	if (isSurveyValid)
    	{
    		if (rootObj.contains(seismicKey0) && rootObj.value(seismicKey0).isArray())
    		{
    			QJsonArray array = rootObj.value(seismicKey0).toArray();
   	    	    for (int i=0; i<array.count(); i++)
   	    	    {
   	    	    	QString txt = array[i].toString("");
   	    	    	std::size_t searchIdx = 0;
        			while (searchIdx<m_seismicManager->listwidgetList->count() && txt.compare(m_seismicManager->listwidgetList->item(searchIdx)->text())!=0)
   	    	    	{
        				searchIdx++;
   	    	    	}
        			if (searchIdx<m_seismicManager->listwidgetList->count())
   	    	    	{
        				m_seismicManager->listwidgetList->item(searchIdx)->setSelected(true);
        				m_seismicManager->f_basketAdd();
   	    	    	}
    	    	}
    		}
    	}

    	if (isSurveyValid)
    	{
    		if (rootObj.contains(horizonKey0) && rootObj.value(horizonKey0).isArray())
    	    {
    			QJsonArray array = rootObj.value(horizonKey0).toArray();
    			for (int i=0; i<array.count(); i++)
    			{
    				QString txt = array[i].toString("");
    				std::size_t searchIdx = 0;
    				while (searchIdx<m_horizonManager->listwidgetList->count() && txt.compare(m_horizonManager->listwidgetList->item(searchIdx)->text())!=0)
    	   	    	{
    					searchIdx++;
    	   	    	}
    				if (searchIdx<m_horizonManager->listwidgetList->count())
    				{
    					m_horizonManager->listwidgetList->item(searchIdx)->setSelected(true);
    					m_horizonManager->f_basketAdd();
    				}
    			}
    	    }
    	}

    	if (isProjectValid)
    	{
    		if (rootObj.contains(culturalKey0) && rootObj.value(culturalKey0).isArray())
    		{
    			QJsonArray array = rootObj.value(culturalKey0).toArray();
    			for (int i=0; i<array.count(); i++)
    			{
    				QString txt = array[i].toString("");
    				std::size_t searchIdx = 0;
    				while (searchIdx<m_culturalsManager->listwidgetList->count() && txt.compare(m_culturalsManager->listwidgetList->item(searchIdx)->text())!=0)
    	    	   	{
    					searchIdx++;
    	    	   	}
    				if (searchIdx<m_culturalsManager->listwidgetList->count())
    				{
    					m_culturalsManager->listwidgetList->item(searchIdx)->setSelected(true);
    					m_culturalsManager->f_basketAdd();
    				}
    			}
    		}

        	// PICKS
        	if (rootObj.contains(picksNamesKey0) && rootObj.value(picksNamesKey0).isArray() &&
        			rootObj.contains(picksPathKey0) && rootObj.value(picksPathKey0).isArray()) {
        		std::vector<QString> tiny;
        		std::vector<QString> full;
        		QJsonArray arrayNames = rootObj.value(picksNamesKey0).toArray();
        		QJsonArray arrayPath = rootObj.value(picksPathKey0).toArray();
        		for (std::size_t i=0; i<arrayNames.count(); i++) {
        			tiny.push_back(arrayNames[i].toString(""));
        			full.push_back(arrayPath[i].toString(""));
        			ProjectManagerNames p;
        			p.copy(tiny, full);
        			m_picksManager->setBasketNames(p);
        			m_picksManager->displayNamesBasket();
        		}
        	}
    	}

        if (isProjectValid)
        {
        	if (rootObj.contains(wellKey0) && rootObj.value(wellKey0).isArray())
        	{
        		QJsonArray array = rootObj.value(wellKey0).toArray();
        		for (std::size_t i=0; i<array.count(); i++)
        		{
        			if (!array[i].isObject())
        			{
        				continue;
        			}
        			QJsonObject wellObj = array[i].toObject();
        			if (!wellObj.contains(wellKey0) || !wellObj.value(wellKey0).isString() || !wellObj.contains(wellLogKey0) || !wellObj.value(wellLogKey0).isArray())
        			{
        				continue;
        			}
        			std::size_t searchIdx = 0;
        			if (!wellObj.contains(wellPathKey0) || !wellObj.value(wellPathKey0).isString() ) {
        				QString txt = wellObj.value(wellKey0).toString("");
        				while (searchIdx<m_wellsManager->m_wellsHeadManager->listwidgetList->count() && txt.compare(m_wellsManager->m_wellsHeadManager->listwidgetList->item(searchIdx)->text())!=0)
        				{
        					searchIdx++;
        				}
        			} else {
        				QString txt = wellObj.value(wellPathKey0).toString("");
        				while (searchIdx<m_wellsManager->m_wellsHeadManager->listwidgetList->count() && txt.compare(m_wellsManager->m_wellsHeadManager->listwidgetList->item(searchIdx)->data(Qt::UserRole).toString())!=0)
        				{
        					searchIdx++;
        				}
        			}
        			if (searchIdx<m_wellsManager->m_wellsHeadManager->listwidgetList->count())
        			{
        				m_wellsManager->m_wellsHeadManager->listwidgetList->item(searchIdx)->setSelected(true);
        				m_wellsManager->m_wellsHeadManager->trt_basketAdd();
        			}
        		}

   	    		for (std::size_t i=0; i<array.count(); i++)
   	    		{
   	    			if (!array[i].isObject())
   	    			{
   	    				continue;
   	    			}
   	    			m_wellsManager->m_wellsHeadManager->listwidgetBasket->clearSelection();

   	    			QJsonObject wellObj = array[i].toObject();
   	    			QString wellbore = wellObj[wellKey0].toString();
   	    			std::size_t searchIdx = 0;
   	    			QString txt = wellObj.value(wellKey0).toString("");
   	    			while (searchIdx<m_wellsManager->m_wellsHeadManager->listwidgetBasket->count() && txt.compare(m_wellsManager->m_wellsHeadManager->listwidgetBasket->item(searchIdx)->text())!=0)
   	    			{
   	    				searchIdx++;
   	    			}
   	    			if (searchIdx<m_wellsManager->m_wellsHeadManager->listwidgetBasket->count())
   	    			{
   	    				m_wellsManager->m_wellsHeadManager->listwidgetBasket->item(searchIdx)->setSelected(true);
   	    				m_wellsManager->m_wellsHeadManager->trt_basketListClick(m_wellsManager->m_wellsHeadManager->listwidgetBasket->item(searchIdx));
   	    			}

   	    			QJsonArray welllog = wellObj[wellLogKey0].toArray();
        	    	int N = welllog.size();
        	    	for (int ii=0; ii<N; ii++)
        	    	{
        	    		QJsonObject obj1 = welllog[ii].toObject();
        	    		int idx_well = -1, idx_bore = -1;
        	    		QString fullname = obj1["fullname"].toString();
        	    		QString tinyname = obj1["tinyname"].toString();
        	    		std::size_t searchIdx = 0;
        	    		txt = tinyname;
        	    		while (searchIdx<m_wellsManager->m_wellsLogManager->listwidgetList->count() && txt.compare(m_wellsManager->m_wellsLogManager->listwidgetList->item(searchIdx)->text())!=0)
        	    		{
        	    			searchIdx++;
        	    		}
        	    		if (searchIdx<m_wellsManager->m_wellsLogManager->listwidgetList->count())
        	    		{
        	    			m_wellsManager->m_wellsLogManager->listwidgetList->item(searchIdx)->setSelected(true);
        	    			m_wellsManager->m_wellsLogManager->f_basketAdd();
        	    		}
        	    	}

        	    	QJsonArray welltf2p = wellObj[wellTFPKey0].toArray();
        	    	N = welltf2p.size();
        	    	m_wellsManager->m_wellsTF2PManager->clearBasket();
        	    	for (int ii=0; ii<N; ii++)
        	    	{
        	    		QJsonObject obj1 = welltf2p[ii].toObject();
        	    	    int idx_well = -1, idx_bore = -1;
        	    	    QString fullname = obj1["fullname"].toString();
        	    	    QString tinyname = obj1["tinyname"].toString();
        	    	    std::size_t searchIdx = 0;
        	    	    txt = tinyname;
        	    	    while (searchIdx<m_wellsManager->m_wellsTF2PManager->listwidgetList->count() && txt.compare(m_wellsManager->m_wellsTF2PManager->listwidgetList->item(searchIdx)->text())!=0)
        	    	    {
        	    	    	searchIdx++;
        	    	    }
        	    	    if (searchIdx<m_wellsManager->m_wellsTF2PManager->listwidgetList->count())
        	    	    {
        	    	    	m_wellsManager->m_wellsTF2PManager->listwidgetList->item(searchIdx)->setSelected(true);
        	    	    	m_wellsManager->m_wellsTF2PManager->f_basketAdd();
        	    	    }
        	    	}

//        	    	QJsonArray wellpicks = wellObj[wellPickKey0].toArray();
//        	    	N = wellpicks.size();
//        	    	for (int ii=0; ii<N; ii++)
//        	    	{
//        	    		QJsonObject obj1 = wellpicks[ii].toObject();
//        	    		int idx_well = -1, idx_bore = -1;
//        	    		QString fullname = obj1["fullname"].toString();
//        	    		QString tinyname = obj1["tinyname"].toString();
//        	    		std::size_t searchIdx = 0;
//        	    		txt = tinyname;
//        	    		while (searchIdx<m_wellsManager->m_wellsPicksManager->listwidgetList->count() && txt.compare(m_wellsManager->m_wellsPicksManager->listwidgetList->item(searchIdx)->text())!=0)
//        	    		{
//        	    			searchIdx++;
//        	    		}
//        	    		if (searchIdx<m_wellsManager->m_wellsPicksManager->listwidgetList->count())
//        	    		{
//        	    			m_wellsManager->m_wellsPicksManager->listwidgetList->item(searchIdx)->setSelected(true);
//        	    			m_wellsManager->m_wellsPicksManager->f_basketAdd();
//        	    		}
//        	    	}
   	    		}
        	}
        }
    }
}


void ProjectManagerWidget::seimsicDatabaseUpdate()
{
	m_seismicManager->f_dataBaseUpdate();
}

void ProjectManagerWidget::wellDatabaseUpdate()
{
	m_wellsManager->trt_dataBaseUpdate0();
}

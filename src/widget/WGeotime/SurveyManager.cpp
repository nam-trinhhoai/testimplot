
#include <QDebug>

#include <SurveyManager.h>

SurveyManager::SurveyManager(QWidget* parent) :
		QWidget(parent) {

	setSeismicManager(nullptr);
	setHorizonManager(nullptr);
	setRgbRawManager(nullptr);

	QGroupBox *groupbox1= new QGroupBox("survey");
	QVBoxLayout* layout1  = new QVBoxLayout(groupbox1);

	lineeditSurveySearch = new QLineEdit;
	listwidgetSurvey = new QListWidget;
	layout1->addWidget(lineeditSurveySearch);
	layout1->addWidget(listwidgetSurvey);


	QVBoxLayout* mainLayout = new QVBoxLayout(this);//(qgb_projectmanager);
	mainLayout->addWidget(groupbox1);

    connect(listwidgetSurvey, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_surveylistClick(QListWidgetItem*)));
	connect(lineeditSurveySearch, SIGNAL(textChanged(QString)), this, SLOT(trt_SearchChange(QString)));
}


SurveyManager::~SurveyManager()
{


}

QString SurveyManager::getName()
{
	std::pair<QString, QString> names = getSurveyPath();
	return names.first;
}


QString SurveyManager::getPath()
{
	std::pair<QString, QString> names = getSurveyPath();
	return names.second;
}

bool SurveyManager::setForceName(QString name)
{
	int idx = 0, N = listwidgetSurvey->count(), cont = 1;
	while ( cont )
	{
		if ( listwidgetSurvey->item(idx)->text().compare(name) != 0 )
			idx++;
		else
			cont = 0;
		if ( idx >= N )
			cont = 0;
	}
	bool res = idx < N;
	if ( res )
	{
		listwidgetSurvey->setCurrentRow(idx);
		trt_surveylistClick(listwidgetSurvey->item(idx));
	}
	return res;
}

void SurveyManager::setSeismicManager(SeismicManager *seismicManager)
{
	m_seismicManager = seismicManager;
}

void SurveyManager::setHorizonManager(HorizonManager *horizonManager)
{
	m_horizonManager = horizonManager;
}

void SurveyManager::setRgbRawManager(RgbRawManager *rgbRawManager)
{
	m_rgbRawManager = rgbRawManager;
}

void SurveyManager::trt_SearchChange(QString str)
{
	updateDisplay();
}

void SurveyManager::trt_surveylistClick(QListWidgetItem*item)
{
	std::pair<QString, QString> names = getSurveyPath();
	if ( m_seismicManager )
	{
		m_seismicManager->setProjectType(m_idxProjectType);
		m_seismicManager->setProjectName(m_projectName);
		qDebug() << "[project name]" << m_projectName << " " << names.first;
		m_seismicManager->setSurveyPath(names.second, names.first);
		qDebug() << "[ survey path ]" << names.first << "   -  " << names.second;
	}
	if ( m_horizonManager )
	{
		m_horizonManager->setSurveyPath(names.second, names.first);
	}
	if ( m_rgbRawManager )
	{
		m_rgbRawManager->setSurvey(names.second, names.first);
	}
	emit surveyChanged();
}


void SurveyManager::setProjectSelectedPath(int projectType, QString path, QString name)
{
	if (m_seismicManager) {
		m_seismicManager->dataClear();
		m_seismicManager->dataBasketClear();
	}
	if (m_horizonManager) {
		m_horizonManager->dataClear();
		m_horizonManager->dataBasketClear();
	}
	if (m_rgbRawManager) {
		m_rgbRawManager->dataClear();
		m_rgbRawManager->dataBasketClear();
	}

	m_projectFullPath = path;
	m_projectName = name;
	m_idxProjectType = projectType;
	// qDebug() << "[ SURVEY: ] " << "name: " << name;
	// qDebug() << "[ SURVEY: ] " << "path: " << path;
	updateNames();
	updateDisplay();
	emit surveyChanged();
}

void SurveyManager::updateDisplay()
{
	std::vector<QString> tiny = m_names.getTiny();
	QString prefix = lineeditSurveySearch->text();
	listwidgetSurvey->clear();
	for (int n=0; n<tiny.size(); n++)
	{
		QString str = tiny[n];
		if ( ProjectManagerNames::isMultiKeyInside(str, prefix ) )
			listwidgetSurvey->addItem(str);
	}
}


std::vector<QString> SurveyManager::getDirList(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
    dir.setSorting(QDir::Name);
    QFileInfoList list = dir.entryInfoList();

    int N = list.size();
    std::vector<QString> listDir;
    listDir.resize(N);
    for (int n=0; n<N; n++)
    {
    	listDir[n] = list[n].absoluteFilePath() + "/";
    }
    return listDir;
}


void SurveyManager::updateNames()
{
	std::vector<QString> fullName = getDirList(m_projectFullPath + surveyRootSubDir);
	std::vector<QString> names = ProjectManagerNames::getNamesFromFullPath(fullName);

	int N = fullName.size();
	for ( int n=0; n<N; n++ )
	{
		QString descFilename = fullName[n] + names[n] + ".desc";
		QString descName = ProjectManagerNames::getKeyFromFilename(descFilename, "name=");
		if ( descName.compare("") != 0 )
		{
			names[n] = descName;
		}
	}
	m_names.copy(names, fullName);
}

std::pair<QString, QString> SurveyManager::getSurveyPath()
{
	QString name = "";
	QString path = "";
	if ( listwidgetSurvey->item(listwidgetSurvey->currentRow()) != nullptr )
	{
		std::vector<QString> tiny = m_names.getTiny();
		std::vector<QString> full = m_names.getFull();
		name = listwidgetSurvey->item(listwidgetSurvey->currentRow())->text();
		int idx = ProjectManagerNames::getIndexFromVectorString(tiny, name);
		path = full[idx];
		// qDebug() << "[SURVEY PATH]" << name << "  -  " << path;
	}
	return std::make_pair(name, path);
}






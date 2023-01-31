
 #include <QDebug>


#include <ProjectManager.h>
#include "globalconfig.h"

ProjectManager::ProjectManager(QWidget* parent) :
		QWidget(parent) {

	// init vectors related to dirProjects
	GlobalConfig& config = GlobalConfig::getConfig();
	const std::vector<std::pair<QString, QString>>& dirProjects = config.dirProjects();
	project_path_nbre = dirProjects.size()+2;
	arrayProjectTypeNames << "None";
	arrayProjectPath << "";

	for (const std::pair<QString, QString>& pair : dirProjects) {
		arrayProjectTypeNames << pair.first;
		arrayProjectPath << pair.second;
	}

	arrayProjectTypeNames << "USER";
	arrayProjectPath << "";

	projectListTinyName.resize(project_path_nbre);
	projectListFullName.resize(project_path_nbre);
	cacheNames.resize(project_path_nbre, nullptr);

	setSurveyManager(nullptr);
	setCulturalsManager(nullptr);
	setWellsManager(nullptr);

	names = new ProjectManagerNames();
	for (int i=0; i<project_path_nbre; i++)
		cacheNames[i] = new ProjectManagerNames();

	QGroupBox *groupbox1= new QGroupBox("project");
	groupbox1->setMinimumHeight(300);
	QVBoxLayout* layout1  = new QVBoxLayout(groupbox1);
	comboboxProjectType = new QComboBox;
	for (int i=0; i<project_path_nbre; i++)
	{
		comboboxProjectType->addItem(arrayProjectTypeNames[i]);
	}

	QHBoxLayout *layoutUserProjectPath = new QHBoxLayout;
	labelUserProjectPath = new QLabel("path:");
	lineeditUserProjectPath = new QLineEdit;
	pushbuttonUserProjectPathValid = new QPushButton("ok");
	layoutUserProjectPath->addWidget(labelUserProjectPath);
	layoutUserProjectPath->addWidget(lineeditUserProjectPath);
	layoutUserProjectPath->addWidget(pushbuttonUserProjectPathValid);
	lineeditProjectSearch = new QLineEdit;
	listwidgetProject = new QListWidget;

	layout1->addWidget(comboboxProjectType);
	layout1->addLayout(layoutUserProjectPath);
	layout1->addWidget(lineeditProjectSearch);
	layout1->addWidget(listwidgetProject);

	QVBoxLayout* mainLayout = new QVBoxLayout(this);//(qgb_projectmanager);
	mainLayout->addWidget(groupbox1);

    connect(comboboxProjectType, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_projecttypeclick(int)));
    connect(listwidgetProject, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_projetlistClick(QListWidgetItem*)));
    connect(lineeditProjectSearch, SIGNAL(textChanged(QString)), this, SLOT(trt_projectSearChchange(QString)));
	connect(pushbuttonUserProjectPathValid, SIGNAL(clicked()), this, SLOT(trt_userProjectValid()));

    projectTypeEnable();
}


ProjectManager::~ProjectManager()
{
	if ( names != nullptr ) delete names;
	for (int i=0; i<project_path_nbre; i++)
	{
		if ( cacheNames[i] != nullptr )
			delete cacheNames[i];
	}
}


QString ProjectManager::getType()
{
	return comboboxProjectType->currentText();

}


QString ProjectManager::getName()
{
	if ( listwidgetProject->item(listwidgetProject->currentRow()) != nullptr )
		return listwidgetProject->item(listwidgetProject->currentRow())->text();
	else
		return "";
}


QString ProjectManager::getPath()
{
	return getProjectFullPath();
}


bool ProjectManager::setProjectType(int idx)
{
	bool res = idx>=0 && idx<comboboxProjectType->count();
	if (res) {
		comboboxProjectType->setCurrentIndex(idx);
		projectTypeEnable();
		trt_projecttypeclick(idx);
	}
	return res;
}

bool ProjectManager::setProjectType(QString projectDirName) {
	int idx = 0, N = comboboxProjectType->count(), cont = 1;
	if (idx>=N) {
		cont = 0;
	}
	while ( cont )
	{
		if ( comboboxProjectType->itemText(idx).compare(projectDirName) != 0 )
			idx++;
		else
			cont = 0;
		if ( idx >= N )
			cont = 0;
	}
	bool res = idx < N;
	if ( res )
	{
		comboboxProjectType->setCurrentIndex(idx);
		projectTypeEnable();
		trt_projecttypeclick(idx);
	}
	return res;
}

bool ProjectManager::setProjectName(QString name)
{
	// QString name = listwidgetProject->item(listwidgetProject->currentRow())->text();
	int idx = 0, N = listwidgetProject->count(), cont = 1;
	if (idx>=N) {
		cont = 0;
	}
	while ( cont )
	{
		if ( listwidgetProject->item(idx)->text().compare(name) != 0 )
			idx++;
		else
			cont = 0;
		if ( idx >= N )
			cont = 0;
	}
	bool res = idx < N;
	if ( res )
	{
		listwidgetProject->setCurrentRow(idx);
		trt_projetlistClick(listwidgetProject->item(idx));
	}
	return res;
}

void ProjectManager::setUserProjectPath(QString path)
{
	lineeditUserProjectPath->setText(path);
}

void ProjectManager::setSurveyManager(SurveyManager *surveyManager)
{
	m_surveyManager = surveyManager;
}

void ProjectManager::setCulturalsManager(CulturalsManager *culturalsManager)
{
	m_culturalsManager = culturalsManager;
}

void ProjectManager::setWellsManager(WellsManager *wellsManager)
{
	m_wellsManager = wellsManager;
}

void ProjectManager::setPicksManager(PicksManager *picksManager)
{
	m_picksManager = picksManager;
}

void ProjectManager::trt_projecttypeclick(int idx)
{
    projectTypeEnable();
	updateNames(idx);
	displayNames();
	emit projectChanged();
}


void ProjectManager::trt_projetlistClick(QListWidgetItem*item)
{
	QString path = getProjectFullPath();
	QString name = listwidgetProject->item(listwidgetProject->currentRow())->text();
	int idx = comboboxProjectType->currentIndex();
	if ( m_surveyManager )
	{
		m_surveyManager->setProjectSelectedPath(idx, path, name);
	}
	if ( m_culturalsManager )
	{
		m_culturalsManager->setProjectType(idx);
		m_culturalsManager->setProjectName(name);
		m_culturalsManager->setProjectPath(path, name);
	}
	if ( m_wellsManager )
	{
		m_wellsManager->setProjectType(idx);
		m_wellsManager->setProjectPath(path, name);
	}
	if ( m_picksManager )
	{
		m_picksManager->setProjectType(idx);
		m_picksManager->setProjectName(name);
		m_picksManager->setProjectPath(path, name);
	}
	emit projectChanged();
}

void ProjectManager::trt_projectSearChchange(QString str)
{
	displayNames();
}


void ProjectManager::setCallBackListClick(void (*ptr)(QListWidgetItem*))
{
	connect(listwidgetProject, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(ptr));
}


QString ProjectManager::getRootProjectPath(int idx)
{
	if ( idx < project_path_nbre-1 )
		return arrayProjectPath[idx];
	else
		return lineeditUserProjectPath->text() + "/";
}

QString ProjectManager::getProjectFullPath()
{
	int idxType = comboboxProjectType->currentIndex();
	QString name = getName();
	if ( name.compare("") == 0 )
		return "";
	else
	{
		// qDebug() << "[ PROJECT CLICK ]:" << getRootProjectPath(idxType) + name + "/";
		return getRootProjectPath(idxType) + listwidgetProject->item(listwidgetProject->currentRow())->text() + "/";
	}
}


std::vector<QString> ProjectManager::getListDir(QString path)
{
    std::vector<QString> tab;
    FILE *pfile;

    char buff[10000];
    fprintf(stderr, "path: %s\n",  path.toStdString().c_str());
    sprintf(buff, "find %s -mindepth 1 -maxdepth 1 \\( -type d -o -type l \\) -exec basename \\{} \\; | sort", path.toStdString().c_str());
    if ( ( pfile = popen(buff, "r") ) == NULL )
    {
        fprintf (stderr, "erreur");
        return tab;
    }
    while (fgets (buff, sizeof(buff), pfile) != NULL)
    {
        strtok(buff, "\n");
        QString txt = QString(buff);
        if ( txt.compare(QString("")) !=0 && txt.compare(QString("."))) tab.push_back(txt);
    }
    fclose(pfile);
    return tab;
}


void ProjectManager::updateNames(int idx)
{
	if ( idx < project_path_nbre-1 )
	{
		if ( cacheNames[idx]->isEmpty() )
		{
			QString path = getRootProjectPath(idx);
			std::vector<QString> listTiny = getListDir(path);
			std::vector<QString> listFull;
			listFull.resize(listTiny.size());
			for (int n=0; n<listTiny.size(); n++)
			{
				listFull[n] = path + listTiny[n];
			}
			cacheNames[idx]->copy(listTiny, listFull);
		}
		names->copy(cacheNames[idx]->getTiny(), cacheNames[idx]->getFull());
	}
	else
	{
		QString path = lineeditUserProjectPath->text() + "/";
		std::vector<QString> listTiny = getListDir(path);
		std::vector<QString> listFull;
		listFull.resize(listTiny.size());
		for (int n=0; n<listTiny.size(); n++)
		{
			listFull[n] = path + listTiny[n];
		}
		names->copy(listTiny, listFull);
	}
}

void ProjectManager::displayNames()
{
	std::vector<QString> listTiny = names->getTiny();
	QString prefix = lineeditProjectSearch->text();
	this->listwidgetProject->clear();
	for (int i=0; i<listTiny.size(); i++)
	{
		QString str = listTiny[i];
		if ( ProjectManagerNames::isMultiKeyInside(str, prefix ) )
	        	this->listwidgetProject->addItem(str);
	    }
}



void ProjectManager::projectTypeEnable()
{
	QString type = getType();
	if ( type.compare("USER") == 0 )
	{
		lineeditUserProjectPath->setEnabled(true);
		pushbuttonUserProjectPathValid->setEnabled(true);
		labelUserProjectPath->setEnabled(true);
	}
	else
	{
		lineeditUserProjectPath->setEnabled(false);
		pushbuttonUserProjectPathValid->setEnabled(false);
		labelUserProjectPath->setEnabled(false);
	}
}


void ProjectManager::trt_userProjectValid()
{
	int idx = project_path_nbre - 1;
	updateNames(idx);
	displayNames();
	emit projectChanged();
}

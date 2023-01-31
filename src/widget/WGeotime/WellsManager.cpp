
#include <QDebug>
#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <WellsManager.h>
#include <wellsDatabaseManager.h>
#include "globalconfig.h"


WellsManager::WellsManager(QWidget* parent) :
		QWidget(parent) {



	QVBoxLayout *mainLayout00 = new QVBoxLayout(this);
	m_wellsHeadManager = new WellsHeadManager;

	QGroupBox *qgb_wellslogs = new QGroupBox();
	QVBoxLayout *wellslogsLayout = new QVBoxLayout(qgb_wellslogs);
	m_wellsLogManager = new WellsLogManager();
	m_wellsLogManager->setWellManager(this);
	// m_projectManager->setCulturalsManager(m_culturalsManager);
	wellslogsLayout->addWidget(m_wellsLogManager);

	// QGroupBox *qgb_wellsPicks = new QGroupBox();
	// QVBoxLayout *wellsPicksLayout = new QVBoxLayout(qgb_wellsPicks);
	// m_wellsPicksManager = new WellsPicksManager();
	// m_wellsPicksManager->setWellManager(this);

		// m_projectManager->setCulturalsManager(m_culturalsManager);
	// wellsPicksLayout->addWidget(m_wellsPicksManager);

	QGroupBox *qgb_wellsTF2P = new QGroupBox();
	QVBoxLayout *wellsTF2PLayout = new QVBoxLayout(qgb_wellsTF2P);
	m_wellsTF2PManager = new WellsTF2PManager();
	m_wellsTF2PManager->setWellManager(this);
		// m_projectManager->setCulturalsManager(m_culturalsManager);
	wellsTF2PLayout->addWidget(m_wellsTF2PManager);

	m_wellsHeadManager->setWellsLogManager(m_wellsLogManager);
	m_wellsHeadManager->setWellsTF2PManager(m_wellsTF2PManager);
	// m_wellsHeadManager-> setWellsPicksManager(m_wellsPicksManager);


	tabwidget = new QTabWidget();
	tabwidget->insertTab(0, qgb_wellslogs, QIcon(QString("")), "Log");
	// tabwidget->insertTab(1, qgb_wellsPicks, QIcon(QString("")), "Picks");
	tabwidget->insertTab(1, qgb_wellsTF2P, QIcon(QString("")), "TF2P");

	QPushButton *pushbutton_dataBaseUpdate0 = new QPushButton("DataBase Update");
	// QPushButton *pushbutton_debug = new QPushButton("get all names");

	connect(pushbutton_dataBaseUpdate0, SIGNAL(clicked()), this, SLOT(trt_dataBaseUpdate0()));
	// connect(pushbutton_debug, SIGNAL(clicked()), this, SLOT(trt_debug()));

	mainLayout00->addWidget(m_wellsHeadManager);
	mainLayout00->addWidget(tabwidget);
	mainLayout00->addWidget(pushbutton_dataBaseUpdate0);
	// mainLayout00->addWidget(pushbutton_debug);
}

WellsManager::~WellsManager()
{

}

void WellsManager::setProjectPath(QString path, QString name)
{

	m_wellsHeadManager->dataClear();
	m_wellsHeadManager->dataBasketClear();

	m_wellsLogManager->dataClear();
	m_wellsLogManager->dataBasketClear();

	// m_wellsPicksManager->dataClear();
	// m_wellsPicksManager->dataBasketClear();

	m_wellsTF2PManager->dataClear();
	// m_wellsPicksManager->dataBasketClear();

	m_wellsHeadManager->m_wellMaster.m_basketWell.clear();
	m_wellsHeadManager->m_wellMaster.m_basketWellBore.clear();
	m_wellsHeadManager->m_wellMaster.finalWellBasket.clear();



	qDebug() << path << " - " << name;
	m_projectname = name;
	m_projectPath = path;
	m_wellPath = m_projectPath + wellsRootSubDir;
	updateNames();
	m_wellsHeadManager->setWellPath(m_wellPath);
	m_wellsHeadManager->setListData(data0);
	// updateNames();
	wellsHeadDisplayNames();
}

void WellsManager::setProjectType(int type)
{
	m_projectType = type;
}

void WellsManager::setForceDataBasket(WELLMASTER _data0)
{
	m_wellsHeadManager->setForceDataBasket(_data0);
}


WELLMASTER WellsManager::getBasket()
{
	// qDebug() << QString::number(m_wellsHeadManager->m_wellMaster.finalWellBasket.size());
	return m_wellsHeadManager->m_wellMaster;
}


std::vector<WELLHEADDATA> WellsManager::getMainData()
{
	return data0;
}


std::vector<MARKER> WellsManager::getPicksSortedWells(const std::vector<QString>& picksName,
		const std::vector<QBrush>& colors)
{
	std::vector<MARKER> data;
	int N = picksName.size();
	data.resize(N);
	int Nwells = data0.size();
	for (int n=0; n<N; n++)
	{
		QString pname = picksName[n];
		data[n].name = pname;
		data[n].color  = colors[n].color();
		for (int n2=0; n2<Nwells; n2++)
		{
			WELLPICKSLIST wellList;
			wellList.name = data0[n2].tinyName;
			wellList.path = data0[n2].fullName;

			for (int nb=0; nb<data0[n2].bore.size(); nb++)
			{
				std::vector<QString> tiny = data0[n2].bore[nb].picks.getTiny();
				std::vector<QString> full = data0[n2].bore[nb].picks.getFull();
				for (int i=0; i<tiny.size(); i++)
				{
					if ( pname.compare(tiny[i]) == 0 )
					{
						WELLBOREPICKSLIST wellBore;
						wellBore.boreName = data0[n2].bore[nb].tinyName;
						wellBore.borePath = data0[n2].bore[nb].fullName;
						wellBore.deviationPath = wellBore.borePath + "/deviation";
						wellBore.picksName = tiny[i];
						wellBore.picksPath = full[i];
						wellList.wellBore.push_back(wellBore);
					}
				}
			}
			if ( wellList.wellBore.size() > 0 )
				data[n].wellPickLists.push_back(wellList);
		}
	}
	return data;
}


void WellsManager::wellsHeadDisplayNames()
{
	m_wellsHeadManager->displayNames();
	/*
	m_wellsHeadManager->listwidgetList->clear();
	for (int n1=0; n1<data0.size(); n1++)
	{
		for (int n2=0; n2<data0[n1].bore.size(); n2++)
		{
			m_wellsHeadManager->listwidgetList->addItem(data0[n1].bore[n2].tinyName);
		}
	}
	*/
}

int WellsManager::getWellsHeadBasketSelectedIndex()
{
	if ( m_wellsHeadManager ) return m_wellsHeadManager->getBasketSelectedIndex();
	return -1;
}

QString WellsManager::getWellsHeadBasketSelectedName()
{
	if ( m_wellsHeadManager ) return m_wellsHeadManager->getBasketSelectedName();
	return "";
}


void WellsManager::updateNames()
{
	QString db_filename = getDatabaseName();
	qDebug() << "[ WELL ] " << db_filename;

	// updateNamesFromDisk(); return;

	if (  QFile::exists(db_filename) )
	{
		updateNamesFromDataBase(db_filename);
	}
	else
	{
		// updateNamesFromDisk();
		// saveListToDataBase(db_filename);
		WellsDatabaseManager::update(db_filename, m_wellPath);
		updateNamesFromDataBase(db_filename);
	}
}


void WellsManager::trt_dataBaseUpdate0()
{
	QString db_filename = getDatabaseName();
	qDebug() << "[ WELL ] " << db_filename;
	// updateNamesFromDisk();
	// saveListToDataBase(db_filename);

	WellsDatabaseManager::update(db_filename, m_wellPath);
	updateNames();

	m_wellsHeadManager->setListData(data0);
	wellsHeadDisplayNames();
}

void WellsManager::updateNamesFromDisk()
{
	qDebug() << m_wellPath;
	QString path = m_wellPath;
	QFileInfoList list = ProjectManagerNames::getDirectoryList(path);
	int N = list.size();
	data0.resize(N);
	for (int n_well=0; n_well<N; n_well++)
	{
		fprintf(stderr, "reading well list: %d / %d\n", n_well+1, N);
		QFileInfo fileInfo = list[n_well];
	    QString filetinyname = fileInfo.fileName();
	    QString filefullname = fileInfo.absoluteFilePath();
	    QDir headDir(filefullname);
	    QString headDescName = filetinyname + ".desc";
	    if (headDir.exists(headDescName)) {
	        QString descFile = headDir.absoluteFilePath(headDescName);
	        QString name = ProjectManagerNames::getKeyTabFromFilename(descFile, "Name");
	        if (!name.isNull() && !name.isEmpty()) {
	            filetinyname = name;
	        }
	    }
	    data0[n_well].tinyName = filetinyname;
	    data0[n_well].fullName = filefullname;

	    QFileInfoList list_bore = ProjectManagerNames::getDirectoryList(filefullname);
	    int Nbores = list_bore.size();
	    data0[n_well].bore.resize(Nbores);

	    for (int n_bore=0; n_bore<Nbores; n_bore++)
		{
		    QFileInfo bore_fileInfo = list_bore[n_bore];
		    QString bore_filetinyname = bore_fileInfo.fileName();
		    QString bore_filefullname = bore_fileInfo.absoluteFilePath();
		    QDir boreDir(bore_filefullname);
		    QString boreDescName = bore_filetinyname + ".desc";
		    if (boreDir.exists(boreDescName)) {
		        QString descFile = boreDir.absoluteFilePath(boreDescName);
		        QString name = ProjectManagerNames::getKeyTabFromFilename(descFile, "Name");
		        if (!name.isNull() && !name.isEmpty()) {
		            bore_filetinyname = name;
		        }
		    }
		    data0[n_well].bore[n_bore].tinyName = QString("[ ") + filetinyname +QString(" ] ") + bore_filetinyname;
		    data0[n_well].bore[n_bore].fullName = bore_filefullname;
		    data0[n_well].bore[n_bore].deviationFullName = getDeviationNames(bore_filefullname);

		    std::pair<std::vector<QString>, std::vector<QString>> tmp;
		    tmp = getLogTF2PPicksNames(bore_filefullname, "*.log");
		    data0[n_well].bore[n_bore].logs.copy(tmp.first, tmp.second);
		    tmp = getLogTF2PPicksNames(bore_filefullname, "*.tfp");
		    data0[n_well].bore[n_bore].tf2p.copy(tmp.first, tmp.second);
		    tmp = getLogTF2PPicksNames(bore_filefullname, "*.pick");
		    data0[n_well].bore[n_bore].picks.copy(tmp.first, tmp.second);
		}
	}
}


void WellsManager::updateNamesFromDataBase(QString db_filename)
{
	fprintf(stderr, "--> %s\n", db_filename.toStdString().c_str());

	data0.clear();
	int nwells = 0;
	char buff[100000], buff2[10000];
	FILE *pFile = NULL;
	pFile = fopen(db_filename.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return;

	fscanf(pFile, "Wells database\n", buff);
	fscanf(pFile, "Wells number: %d\n", &nwells);
	data0.resize(nwells);

	int n0, t1, t2, t3;
	int Nbore, N2;
	std::vector<QString> tiny;
	std::vector<QString> full;
	for (int n=0; n<nwells; n++)
	{
		fscanf(pFile, "head: %d %[^;];%[^\n]\n", &n0, buff, buff2);
		data0[n].tinyName = QString(buff);
		data0[n].fullName = QString(buff2);
		fscanf(pFile, "bore number: %d\n", &Nbore);
		data0[n].bore.resize(Nbore);
		for (int n2=0; n2<Nbore; n2++)
		{
			fscanf(pFile, "head: %d bore:%d %[^;];%[^\n]\n", &t1, &t2, buff, buff2);
			data0[n].bore[n2].tinyName = QString(buff);
			data0[n].bore[n2].fullName = QString(buff2);
			data0[n].bore[n2].deviationFullName = getDeviationNames(data0[n].bore[n2].fullName);

			fscanf(pFile, "logs number: %d\n", &N2);
			tiny.resize(N2);
			full.resize(N2);
			for (int n3=0; n3<N2; n3++)
			{
				fscanf(pFile, "head:%d bore:%d log:%d %[^;];%[^\n]\n", &t1, &t2, &t3, buff, buff2);
				tiny[n3] = QString(buff);
				full[n3] = QString(buff2);
	    	}
			data0[n].bore[n2].logs.copy(tiny, full);

			fscanf(pFile, "tf2p number: %d\n", &N2);
	   		tiny.resize(N2);
	   		full.resize(N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fscanf(pFile, "head:%d bore:%d tf2p:%d %[^;];%[^\n]\n", &t1, &t2, &t3, buff, buff2);
	    		tiny[n3] = QString(buff);
	    		full[n3] = QString(buff2);
	    	}
	    	data0[n].bore[n2].tf2p.copy(tiny, full);

	    	fscanf(pFile, "picks number: %d\n", &N2);
	    	tiny.resize(N2);
	    	full.resize(N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fscanf(pFile, "head:%d bore:%d picks:%d %[^;];%[^\n]\n", &t1, &t2, &t3, buff, buff2);
	    		tiny[n3] = QString(buff);
	    		full[n3] = QString(buff2);
	    	}
	    	data0[n].bore[n2].picks.copy(tiny, full);

	    }
	}
	fclose(pFile);
}


void WellsManager::saveListToDataBase(QString db_filename)
{
	FILE *pFile = NULL;

	fprintf(stderr, "database filename: %s\n", db_filename.toStdString().c_str());
	pFile = fopen(db_filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	fprintf(pFile, "Wells database\n");
	fprintf(pFile, "Wells number: %d\n", data0.size());
	std::vector<QString> tiny;
	std::vector<QString> full;
	for (int n=0; n<data0.size(); n++)
	{
	    fprintf(pFile, "head:%d %s;%s\n", n, data0[n].tinyName.toStdString().c_str(), data0[n].fullName.toStdString().c_str());
	    int Nbore = data0[n].bore.size();
	    fprintf(pFile, "bore number: %d\n", Nbore);
	    for (int n2=0; n2<Nbore; n2++)
	    {
	    	fprintf(pFile, "head: %d bore: %d %s;%s\n", n, n2, data0[n].bore[n2].tinyName.toStdString().c_str(), data0[n].bore[n2].fullName.toStdString().c_str());

	    	tiny = data0[n].bore[n2].logs.getTiny();
	    	full = data0[n].bore[n2].logs.getFull();
	    	int N2 = tiny.size();
	    	fprintf(pFile, "logs number: %d\n", N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fprintf(pFile, "head: %d bore: %d log: %d %s;%s\n", n, n2, n3, tiny[n3].toStdString().c_str(), full[n3].toStdString().c_str());
	    	}

	    	tiny = data0[n].bore[n2].tf2p.getTiny();
	    	full = data0[n].bore[n2].tf2p.getFull();
	    	N2 = tiny.size();
	    	fprintf(pFile, "tf2p number: %d\n", N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fprintf(pFile, "head: %d bore: %d tf2p: %d %s;%s\n", n, n2, n3, tiny[n3].toStdString().c_str(), full[n3].toStdString().c_str());
	    	}

	    	tiny = data0[n].bore[n2].picks.getTiny();
	    	full = data0[n].bore[n2].picks.getFull();
	    	N2 = tiny.size();
	    	fprintf(pFile, "picks number: %d\n", N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fprintf(pFile, "head: %d bore: %d picks: %d %s;%s\n", n, n2, n3, tiny[n3].toStdString().c_str(), full[n3].toStdString().c_str());
	    	}
	    }
	}
	fclose(pFile);
	chmod(db_filename.toStdString().c_str(), (mode_t)0777);
}

void WellsManager::f_dataBaseUpdate()
{
	QString db_filename = getDatabaseName();
	updateNamesFromDisk();
	saveListToDataBase(db_filename);
	wellsHeadDisplayNames();
}

QString WellsManager::getDatabaseName()
{
	// qDebug() << "[project name]" << getProjectName();
	// return getDatabasePath() + "database_wells_" + getProjIndexNameForDataBase() + QString("_") + getProjectName() + ".txt";
	GlobalConfig& config = GlobalConfig::getConfig();
	QString tmp = ObjectManager::formatDirPath(m_wellPath);
	tmp.replace("/", "_@_");
	return config.databasePath() + QString("/database_wells_") + tmp + ".txt";
}





QString WellsManager::getDeviationNames(QString path)
{
	QString filename = path + "/" + deviationFilename;
	if ( QFile::exists(filename) )
	{
		return filename;
	}
	return "";
}


std::pair<std::vector<QString>, std::vector<QString>> WellsManager::getLogTF2PPicksNames(QString path, QString ext)
{
	std::vector<QString> tinyNames;
	std::vector<QString> fullNames;

	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << ext;
	dir.setNameFilters(filters);
	QFileInfoList list = dir.entryInfoList();

	int N = list.size();
	tinyNames.resize(N);
	fullNames.resize(N);

	for (int i=0; i<list.size(); i++)
	{
		QFileInfo fileInfo = list.at(i);
	    QString filename = fileInfo.fileName();
	    fullNames[i] = fileInfo.absoluteFilePath();
	    tinyNames[i] = ProjectManagerNames::getKeyTabFromFilename(fullNames[i], "Name");
	}
	return std::make_pair(tinyNames, fullNames);
}


std::vector<std::vector<QString>> WellsManager::getWellBasketLogPicksTf2pNames(QString type, QString nameType)
{
	if ( m_wellsHeadManager ) return m_wellsHeadManager->getWellBasketLogPicksTf2pNames(type, nameType);
	std::vector<std::vector<QString>> out;
	return out;
}

std::vector<QString> WellsManager::getWellBasketTinyNames()
{
	if ( m_wellsHeadManager ) return m_wellsHeadManager->getWellBasketTinyNames();
	std::vector<QString> out;
	return out;
}

std::vector<QString> WellsManager::getWellBasketFullNames()
{

	if ( m_wellsHeadManager ) return m_wellsHeadManager->getWellBasketFullNames();
	std::vector<QString> out;
	return out;
}


void WellsManager::trt_debug()
{
	/*
	std::vector<std::vector<QString>> data = m_wellsHeadManager->getWellBasketLogPicksTf2pNames("log", "tiny");
	std::vector<QString> wellBasketFull = m_wellsHeadManager->getWellBasketFullNames();
	std::vector<QString> wellBasketTiny = m_wellsHeadManager->getWellBasketTinyNames();

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

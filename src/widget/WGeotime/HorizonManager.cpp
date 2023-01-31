
#include <iostream>
#include <sys/stat.h>
#include <QDebug>
#include <freeHorizonManager.h>
#include <HorizonManager.h>
#include <geotimepath.h>
#include "globalconfig.h"

HorizonManager::HorizonManager(QWidget* parent) :
		ObjectManager(parent) {

}

HorizonManager::~HorizonManager()
{

}

std::vector<QString> HorizonManager::getNames()
{
	return m_namesBasket.getTiny();
}

std::vector<QString> HorizonManager::getPath()
{
	return m_namesBasket.getFull();
}

std::vector<QString> HorizonManager::getAllNames()
{
	return m_names.getTiny();
}

std::vector<QString> HorizonManager::getAllPath()
{
	return m_names.getFull();
}

void HorizonManager::setSurveyPath(QString path, QString name)
{
	dataClear();
	dataBasketClear();
	m_surveyPath = path;
	m_surveyName = name;
	m_horizonPath = m_surveyPath + horizonRootSubDir;
	qDebug() << "[ HORIZON ]" << m_horizonPath;
	updateNames();
	displayNames();
}

void HorizonManager::setForceBasket(std::vector<QString> tinyName0)
{
	clearBasket();
	int N = tinyName0.size();
	int N2 = listwidgetList->count();
	// std::vector<QString> fullName = m_names.getFull();
	// std::vector<QString> tinyName = m_names.getTiny();
	for (int n=0; n<N; n++)
	{
		int cont = 1, idx = 0;
		while ( cont )
		{
			if ( listwidgetList->item(idx)->text().compare(tinyName0[n]) != 0 )
				idx++;
			else
				cont = 0;
			if ( idx >= N2 )
				cont = 0;
		}
		if ( idx < N2 )
		{
			listwidgetList->setCurrentRow(idx);
			f_basketAdd();
		}
	}
}

void HorizonManager::updateNames()
{
	QString db_filename = getDatabaseName();

	updateNamesFromDisk(); return;
	if (  QFile::exists(db_filename) )
	{
		updateNamesFromDataBase(db_filename);
	}
	else
	{
		updateNamesFromDisk();
		saveListToDataBase(db_filename);
	}
}


QString HorizonManager::getDatabaseName()
{
	// qDebug() << "[project name]" << getProjectName();
	// return getDatabasePath() + "database_horizons_" + getProjIndexNameForDataBase() + QString("_") + getProjectName() + QString("_") + getSurveyName() + ".txt";
	QString tmp = formatDirPath(m_horizonPath);
	tmp.replace("/", "_@_");
	GlobalConfig& config = GlobalConfig::getConfig();
	return config.databasePath() + QString("/database_horizon_") + tmp + ".txt";
}

void HorizonManager::updateNamesFromDisk()
{
	QString path = m_horizonPath;
	QDir dir = QDir(path);
	dir.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	dir.setSorting(QDir::Name);
	QFileInfoList list = dir.entryInfoList();

	std::vector<QString> full;
	std::vector<QString> tiny;

    int N = list.size();
    for (int i=0; i<N; i++)
    {
    	QFileInfo fileInfo = list.at(i);
	    QString path1 = path + fileInfo.fileName() + QString("/HORIZON_GRIDS/");
	    fprintf(stderr, "horizons path1 --> %s\n", path1.toStdString().c_str());
	    QDir dir0 = QDir(path1);
	    dir0.setFilter(QDir::Files);
	    // dir->setFilter(QDir::AllEntries );
	    dir0.setSorting(QDir::Name);
	    QStringList filters;
	    filters << "*.raw";
	    dir0.setNameFilters(filters);
	    QFileInfoList list0 = dir0.entryInfoList();
	    int NN = list0.size();
	    for (int ii=0; ii<NN; ii++)
	    {
	    	QFileInfo fileInfo0 = list0.at(ii);
	        QString path0 = fileInfo0.fileName();
	        QString tmp = path1 + path0;
	        full.push_back(tmp);
	        QString rawname = path0.split(".",Qt::SkipEmptyParts).at(0);
	        tiny.push_back(rawname + " (" + fileInfo.fileName() + ")");
	    }
    }
    m_names.copy(tiny, full);
}


void HorizonManager::updateNamesFromDataBase(QString db_filename)
{
	fprintf(stderr, "--> %s\n", db_filename.toStdString().c_str());

	std::vector<QString> full;
	std::vector<QString> tiny;

	int N = 0, n0 = 0;
	char buff[100000], buff2[10000];
	FILE *pFile = NULL;
	pFile = fopen(db_filename.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return;

	fscanf(pFile, "Horizon database\n", buff);
	fscanf(pFile, "Horizon number: %d\n", &N);
	tiny.resize(N);
	full.resize(N);
	for (int n=0; n<N; n++)
	{
		fscanf(pFile, "%d %[^;];%[^\n]\n", &n0, buff, buff2);
		tiny[n] = QString(buff);
		full[n] = QString(buff2);
	}
	fclose(pFile);
	m_names.copy(tiny, full);
}


void HorizonManager::saveListToDataBase(QString db_filename)
{
	FILE *pFile = NULL;
	fprintf(stderr, "database filename: %s\n", db_filename.toStdString().c_str());
	pFile = fopen(db_filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	std::vector<QString> full = m_names.getFull();
	std::vector<QString> tiny = m_names.getTiny();


	int N = tiny.size();
	fprintf(pFile, "Horizon database\n");
	fprintf(pFile, "Horizon number: %d\n", N);
	for (int n=0; n<N; n++)
	{
		fprintf(pFile, "%d %s;%s\n", n, tiny[n].toStdString().c_str(), full[n].toStdString().c_str());
	}
	fclose(pFile);
	chmod(db_filename.toStdString().c_str(), (mode_t)0777);
}

void HorizonManager::f_dataBaseUpdate()
{
	QString db_filename = getDatabaseName();
	updateNamesFromDisk();
	saveListToDataBase(db_filename);
	displayNames();
}

// ================================================
std::vector<QString> HorizonManager::getIsoValueListName()
{
	std::vector<QString> out;
	QString path = m_surveyPath + "/ImportExport/IJK/HORIZONS/ISOVAL/";
	QDir mainDir(path);
	if ( !mainDir.exists() ) return out;
	QFileInfoList mainList = mainDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	if ( mainList.size() == 0 ) return out;
	out.resize(mainList.size());

	for (int i=0; i<mainList.size(); i++)
	{
		out[i] = mainList[i].fileName();
	}
	return out;
}

std::vector<QString> HorizonManager::getIsoValueListPath()
{
	std::vector<QString> out;
	QString path = m_surveyPath + "/ImportExport/IJK/HORIZONS/ISOVAL/";
	QDir mainDir(path);
	if ( !mainDir.exists() ) return out;
	QFileInfoList mainList = mainDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	if ( mainList.size() == 0 ) return out;
	out.resize(mainList.size());

	for (int i=0; i<mainList.size(); i++)
	{
		out[i] = path + mainList[i].fileName();
	}
	return out;
}

std::vector<QString> HorizonManager::getFreeName()
{
	std::vector<QString> out;
	// QString path = m_surveyPath + "/ImportExport/IJK/HORIZONS/" + QString::fromStdString(FreeHorizonManager::BaseDirectory) + "/";
	QString path = m_surveyPath + "/" + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_PATH) + "/";
	QFileInfoList infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].baseName());
	return out;
}

std::vector<QString> HorizonManager::getFreePath()
{
	std::vector<QString> out;
	// QString path = m_surveyPath + "/ImportExport/IJK/HORIZONS/" + QString::fromStdString(FreeHorizonManager::BaseDirectory) + "/";
	QString path = m_surveyPath + "/" + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_PATH) + "/";
	QFileInfoList infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].absoluteFilePath());
	return out;
}


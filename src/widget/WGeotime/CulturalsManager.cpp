#include <QDebug>


#include <iostream>
#include <sys/stat.h>
#include <stdio.h>
#include <QFile>
#include <CulturalsManager.h>

CulturalsManager::CulturalsManager(QWidget* parent)
{

}

CulturalsManager::~CulturalsManager()
{

}

void CulturalsManager::setProjectPath(QString path, QString name)
{
	dataClear();
	dataBasketClear();
	m_cdat.clear();
	m_strd.clear();
	m_cdatBasket.clear();
	m_strdBasket.clear();
	m_projectName = name;
	m_projectPath = path;
	m_culturalsPath = m_projectPath + culturalsRootSubDir;
	updateNames();
	displayNames();
}


std::vector<QString> CulturalsManager::getCdatNames()
{
	std::vector<QString> tiny = m_namesBasket.getTiny();
	std::vector<QString> full = m_namesBasket.getFull();
	std::vector<QString> ret;

	for (int n=0; n<tiny.size(); n++)
	{
		QFileInfo fi(full[n]);
		QString ext = fi.suffix();
		if ( ext.compare("cdat") == 0 )
		{
			ret.push_back(tiny[n]);
		}
	}
	return ret;
}


std::vector<QString> CulturalsManager::getCdatPath()
{
	std::vector<QString> tiny = m_namesBasket.getTiny();
	std::vector<QString> full = m_namesBasket.getFull();
	std::vector<QString> ret;

	for (int n=0; n<tiny.size(); n++)
	{
		QFileInfo fi(full[n]);
		QString ext = fi.suffix();
		if ( ext.compare("cdat") == 0 )
		{
			ret.push_back(full[n]);
		}
	}
	return ret;
}

void CulturalsManager::setForceBasket(std::vector<QString> cdatName, std::vector<QString> cdatPath,
		std::vector<QString> strdName, std::vector<QString> strdPath)
{

	int N = cdatName.size();
	std::vector<QBrush> cdatColor;
	cdatColor.resize(N);
	for (int n=0; n<N; n++)
		cdatColor[n] = Qt::yellow;

	N = strdName.size();
	std::vector<QBrush> strdColor;
	strdColor.resize(N);
	for (int n=0; n<N; n++)
		strdColor[n] = Qt::green;

	m_cdatBasket.copy(cdatName, cdatPath, cdatColor);
	m_strdBasket.copy(strdName, strdPath, strdColor);
	m_namesBasket.copy(cdatName, cdatPath, cdatColor);
	m_namesBasket.add(strdName, strdPath, strdColor);
	displayNamesBasket();
	listwidgetList->clearSelection();
	listwidgetBasket->clearSelection();
}


std::vector<QString> CulturalsManager::getStrdNames()
{
	std::vector<QString> tiny = m_namesBasket.getTiny();
	std::vector<QString> full = m_namesBasket.getFull();
	std::vector<QString> ret;

	for (int n=0; n<tiny.size(); n++)
	{
		QFileInfo fi(full[n]);
		QString ext = fi.suffix();
		if ( ext.compare("strd") == 0 )
		{
			ret.push_back(tiny[n]);
		}
	}
	return ret;
}


std::vector<QString> CulturalsManager::getStrdPath()
{
	std::vector<QString> tiny = m_namesBasket.getTiny();
	std::vector<QString> full = m_namesBasket.getFull();
	std::vector<QString> ret;

	for (int n=0; n<tiny.size(); n++)
	{
		QFileInfo fi(full[n]);
		QString ext = fi.suffix();
		if ( ext.compare("strd") == 0 )
		{
			ret.push_back(full[n]);
		}
	}
	return ret;
}

void CulturalsManager::f_basketAdd()
{
	QList<QListWidgetItem*> list0 = listwidgetList->selectedItems();
	std::vector<QString> cdatTiny = m_cdat.getTiny();
	std::vector<QString> cdatFull = m_cdat.getFull();

	std::vector<QString> strdTiny = m_strd.getTiny();
	std::vector<QString> strdFull = m_strd.getFull();

	std::vector<QString> cdatBasketTiny = m_cdatBasket.getTiny();
	std::vector<QString> cdatBasketFull = m_cdatBasket.getFull();
	std::vector<QBrush> cdatBasketColor = m_cdatBasket.getColor();

	std::vector<QString> strdBasketTiny = m_strdBasket.getTiny();
	std::vector<QString> strdBasketFull = m_strdBasket.getFull();
	std::vector<QBrush> strdBasketColor = m_strdBasket.getColor();

	std::vector<QString> currentCdatBasketFull = m_cdatBasket.getFull();
	std::vector<QString> currentStrdBasketFull = m_strdBasket.getFull();


	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		QBrush brush = list0[i]->foreground();
		if ( brush == Qt::yellow )
		{
			int idx = ProjectManagerNames::getIndexFromVectorString(cdatTiny, txt);
			if ( idx >= 0 )
			{
				int idxBasket = ProjectManagerNames::getIndexFromVectorString(currentCdatBasketFull, cdatFull[idx]);
				if ( idxBasket < 0 )
				{
					cdatBasketTiny.push_back(cdatTiny[idx]);
					cdatBasketFull.push_back(cdatFull[idx]);
					cdatBasketColor.push_back(Qt::yellow);
				}
			}
		}
		else
		{
			int idx = ProjectManagerNames::getIndexFromVectorString(strdTiny, txt);
			if ( idx >= 0 )
			{
				int idxBasket = ProjectManagerNames::getIndexFromVectorString(currentStrdBasketFull, strdFull[idx]);
				if ( idxBasket < 0 )
				{
					strdBasketTiny.push_back(strdTiny[idx]);
					strdBasketFull.push_back(strdFull[idx]);
					strdBasketColor.push_back(Qt::green);
				}
			}
		}
	}
	m_cdatBasket.copy(cdatBasketTiny, cdatBasketFull, cdatBasketColor);
	m_strdBasket.copy(strdBasketTiny, strdBasketFull, strdBasketColor);
	m_namesBasket.copy(cdatBasketTiny, cdatBasketFull, cdatBasketColor);
	m_namesBasket.add(strdBasketTiny, strdBasketFull, strdBasketColor);
	displayNamesBasket();
	listwidgetList->clearSelection();
	listwidgetBasket->clearSelection();
}


void CulturalsManager::f_basketSub()
{
	std::vector<QString> cdatBasketTiny = m_cdatBasket.getTiny();
	std::vector<QString> cdatBasketFull = m_cdatBasket.getFull();
	std::vector<QBrush> cdatBasketColor = m_cdatBasket.getColor();

	std::vector<QString> strdBasketTiny = m_strdBasket.getTiny();
	std::vector<QString> strdBasketFull = m_strdBasket.getFull();
	std::vector<QBrush> strdBasketColor = m_strdBasket.getColor();

	QList<QListWidgetItem*> list0 = listwidgetBasket->selectedItems();
	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		QBrush brush = list0[i]->foreground();
		if ( brush == Qt::yellow )
		{
			int idx = ProjectManagerNames::getIndexFromVectorString(cdatBasketTiny, txt);
			if ( idx >= 0 )
			{
				cdatBasketTiny.erase(cdatBasketTiny.begin()+idx, cdatBasketTiny.begin()+idx+1);
				cdatBasketFull.erase(cdatBasketFull.begin()+idx, cdatBasketFull.begin()+idx+1);
				cdatBasketColor.erase(cdatBasketColor.begin()+idx, cdatBasketColor.begin()+idx+1);
			}
		}
		else
		{
			int idx = ProjectManagerNames::getIndexFromVectorString(strdBasketTiny, txt);
			if ( idx >= 0 )
			{
				strdBasketTiny.erase(strdBasketTiny.begin()+idx, strdBasketTiny.begin()+idx+1);
				strdBasketFull.erase(strdBasketFull.begin()+idx, strdBasketFull.begin()+idx+1);
				strdBasketColor.erase(strdBasketColor.begin()+idx, strdBasketColor.begin()+idx+1);
			}
		}
	}
	m_cdatBasket.copy(cdatBasketTiny, cdatBasketFull, cdatBasketColor);
	m_strdBasket.copy(strdBasketTiny, strdBasketFull, strdBasketColor);
	m_namesBasket.copy(cdatBasketTiny, cdatBasketFull, cdatBasketColor);
	m_namesBasket.add(strdBasketTiny, strdBasketFull, strdBasketColor);
	listwidgetBasket->clearSelection();
	displayNamesBasket();
}



void CulturalsManager::updateNames()
{
	QString db_filename = getDatabaseName();
	qDebug() << "[ CULTURALS ] " << db_filename;
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


void CulturalsManager::updateNamesFromDisk()
{
	m_cdat = updateNames("cdat");
	m_strd = updateNames("strd");
	m_names.copy(m_cdat.getTiny(), m_cdat.getFull(), m_cdat.getColor());
	m_names.add(m_strd.getTiny(), m_strd.getFull(), m_strd.getColor());
}


ProjectManagerNames CulturalsManager::updateNames(QString ext)
{

	QDir dir(m_culturalsPath);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	if ( ext.compare("cdat") == 0 )
		filters << "*.cdat";
	else if ( ext.compare("strd") == 0 )
		filters << "*.strd";
	dir.setNameFilters(filters);
	QFileInfoList list = dir.entryInfoList();
	int N = list.size();
	std::vector<QString> tiny;
	std::vector<QString> full;
	std::vector<QBrush> color;
	tiny.resize(N);
	full.resize(N);
	color.resize(N);
	QBrush color0;
	if ( ext.compare("cdat") == 0 )
		color0 = Qt::yellow;
	else if ( ext.compare("strd") == 0 )
		color0 = Qt::green;
	for (int n=0; n<N; n++)
	{
		QFileInfo fileInfo = list.at(n);
		QString fullname0 = fileInfo.absoluteFilePath();
		QString tiny0 = "";
		QFile inputFile(fullname0);
		if ( inputFile.open(QIODevice::ReadOnly) )
		{
			QTextStream in(&inputFile);
			tiny0 = in.readLine();
			inputFile.close();
		}
		tiny[n] = tiny0;
		full[n] = fullname0;
		color[n] = color0;
	}
	ProjectManagerNames data;
	data.copy(tiny, full, color);
	return data;
}


void CulturalsManager::updateNamesFromDataBase(QString db_filename)
{
	fprintf(stderr, "--> %s\n", db_filename.toStdString().c_str());

	int N = 0;
	char buff[100000], buff2[10000];
	int n0, t1, t2, t3;

	FILE *pFile = NULL;
	pFile = fopen(db_filename.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return;

	std::vector<QString> cdat_full;
	std::vector<QString> cdat_tiny;
	std::vector<QBrush> color;

	fscanf(pFile, "Cultural database\n", buff);
	fscanf(pFile, "cdata number: %d\n", &N);
	cdat_tiny.resize(N);
	cdat_full.resize(N);
	color.resize(N);
	for (int n=0; n<N; n++)
	{
		fscanf(pFile, "%d %[^;];%[^\n]\n", &n0, buff, buff2);
		cdat_tiny[n] = QString(buff);
		cdat_full[n] = QString(buff2);
		color[n] = Qt::yellow;
	}
	m_names.copy(cdat_tiny, cdat_full, color);

	std::vector<QString> strd_full;
	std::vector<QString> strd_tiny;
	fscanf(pFile, "strd number: %d\n", &N);
	strd_tiny.resize(N);
	strd_full.resize(N);
	color.resize(N);
	for (int n=0; n<N; n++)
	{
		fscanf(pFile, "%d %[^;];%[^\n]\n", &n0, buff, buff2);
		strd_tiny[n] = QString(buff);
		strd_full[n] = QString(buff2);
		color[n] = Qt::green;
	}
	m_names.add(strd_tiny, strd_full, color);
	fclose(pFile);
}


void CulturalsManager::saveListToDataBase(QString db_filename)
{
	FILE *pFile = NULL;
	fprintf(stderr, "database filename: %s\n", db_filename.toStdString().c_str());
	pFile = fopen(db_filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	fprintf(pFile, "Cultural database\n");

	std::vector<QString> tiny = m_cdat.getTiny();
	std::vector<QString> full = m_cdat.getFull();

	int N = tiny.size();
	fprintf(pFile, "cdata number: %d\n", N);
	for (int n=0; n<N; n++)
	{
		fprintf(pFile, "%d %s;%s\n", n, tiny[n].toStdString().c_str(), full[n].toStdString().c_str());
   	}

	tiny = m_strd.getTiny();
	full = m_strd.getFull();
	N = tiny.size();
	fprintf(pFile, "strd number: %d\n", N);
	for (int n=0; n<N; n++)
	{
		fprintf(pFile, "%d %s;%s\n", n, tiny[n].toStdString().c_str(), full[n].toStdString().c_str());
	}
	fclose(pFile);
	chmod(db_filename.toStdString().c_str(), (mode_t)0777);
}


QString CulturalsManager::getDatabaseName()
{
	// qDebug() << "[project name]" << getProjectName();
	return getDatabasePath() + "database_culturals_" + getProjIndexNameForDataBase() + QString("_") + getProjectName() + ".txt";
}


/*
void GeotimeProjectManagerWidget::cultural_names_disk_update()
{
	if ( !isTabCulturals() ) return;

	QString path = get_cultural_path0();
	QFileInfoList cdata_list = get_cultural_cdata_list(path);
	QFileInfoList strd_list = get_cultural_strd_list(path);
	int N = cdata_list.size();
	display0.culturals.cdata_tinyname.resize(N);
	display0.culturals.cdata_fullname.resize(N);
	for (int n=0; n<N; n++)
	{
		QFileInfo fileInfo = cdata_list.at(n);
	    QString filename = fileInfo.fileName();
	    QString fullname = path + filename;
	    display0.culturals.cdata_fullname[n] = fullname;
	    QString tinyname = fileInfo.completeBaseName();
	    FILE *pfile = fopen(fullname.toStdString().c_str(), "r");
	    if( pfile )
	    {
	    	char buff[10000];
	        fscanf(pfile, "%s\n", buff);
	        tinyname = QString(buff);
	        fclose(pfile);
	    }
	    display0.culturals.cdata_tinyname[n] = tinyname;
	}

	N = strd_list.size();
	display0.culturals.strd_tinyname.resize(N);
	display0.culturals.strd_fullname.resize(N);
	for (int n=0; n<N; n++)
	{
		QFileInfo fileInfo = strd_list.at(n);
	    QString filename = fileInfo.fileName();
	    QString fullname = path + filename;
	    display0.culturals.strd_fullname[n] = fullname;
	    QString tinyname = fileInfo.completeBaseName();
	    FILE *pfile = fopen(fullname.toStdString().c_str(), "r");
	    if( pfile )
	    {
	    	char buff[10000];
	        fscanf(pfile, "%s\n", buff);
	        tinyname = QString(buff);
	        fclose(pfile);
	    }
	    display0.culturals.strd_tinyname[n] = tinyname;
	}
}
*/

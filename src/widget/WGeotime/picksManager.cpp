#include <stdio.h>

#include <QDebug>
#include <QProcess>
#include "Xt.h"
#include <iostream>
#include <sys/stat.h>
#include <picksmanager.h>
#include "globalconfig.h"


PicksManager::PicksManager(QWidget* parent)
{


}


PicksManager::~PicksManager()
{


}


void PicksManager::setProjectPath(QString path, QString name)
{
	dataClear();
	dataBasketClear();
//	m_picksName.clear();
//	m_picksBasket.clear();
	m_projectName = name;
	m_projectPath = path;
	m_picksPath = m_projectPath + picksRootSubDir;
	updateNames();
	displayNames();
}


std::vector<QString> PicksManager::getNames()
{
	return m_namesBasket.getTiny();
}

std::vector<QString> PicksManager::getPath()
{
	return m_namesBasket.getFull();
}

std::vector<QBrush> PicksManager::getColors() {
	std::vector<QBrush> colors;
	std::vector<QString> paths = getPath();
	std::vector<QString> allPaths = getAllPath();
	colors.resize(paths.size(), QBrush(QColor(0, 0, 0)));
	for (int i=0; i<paths.size(); i++) {
		QString searchPath = paths[i];
		bool notFound = true;
		int j = 0;
		while (notFound && j<allPaths.size()) {
			notFound = searchPath.compare(allPaths[j])!=0;
			if (notFound) {
				j++;
			}
		}
		if (!notFound) {
			colors[i] = m_allColors[j];
		}
	}
	return colors;
}

std::vector<QString> PicksManager::getAllNames()
{
	return m_names.getTiny();
}

std::vector<QString> PicksManager::getAllPath()
{
	return m_names.getFull();
}

std::vector<QBrush> PicksManager::getAllColors() {
	return m_allColors;
}

QString PicksManager::getDatabaseName()
{
	// qDebug() << "[project name]" << getProjectName();
	return getDatabasePath() + "database_picks_" + getProjIndexNameForDataBase() + QString("_") + getProjectName() + ".txt";
}


void PicksManager::updateNames()
{
	QString db_filename = getDatabaseName();
	qDebug() << "[ PICKS ] " << db_filename;
	// updateNamesFromDisk(); return;

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


QString PicksManager::fullFilenameToTinyName(QString filename)
{
	QFileInfo fi(filename);
	QString ext = fi.suffix();
	QString descFilename = filename + "/desc";
	// qDebug() << descFilename;
	QString name0 = ProjectManagerNames::getKeyFromFilename(descFilename, "name=");
	return name0;
}


QColor PicksManager::fullFilenameToColor(QString filename)
{
	QFileInfo fi(filename);
	QString ext = fi.suffix();
	QString descFilename = filename + "/desc";
	// qDebug() << descFilename;
	QString name0 = ProjectManagerNames::getKeyFromFilename(descFilename, "color=");
	bool valid = name0.count()>=11;
	int red = 0;
	int green = 0;
	int blue = 0;
	if (valid) {
		QString redStr = QString(name0[2]) + name0[3] + name0[4];
		red = redStr.toInt(&valid);
	}
	if (valid) {
		QString greenStr = QString(name0[5]) + name0[6] + name0[7];
		green = greenStr.toInt(&valid);
	}
	if (valid) {
		QString blueStr = QString(name0[8]) + name0[9] + name0[10];
		blue = blueStr.toInt(&valid);
	}
	if (!valid) {
		red = 0;
		green = 0;
		blue = 0;
	}
	QColor color(red, green, blue);
	return color;
}

void PicksManager::updateNamesFromDisk()
{
	// qDebug() << m_picksPath;
	QString path = m_picksPath;
	QFileInfoList list = ProjectManagerNames::getDirectoryList(path);
	int N = list.size();
	std::vector<QString> tiny;
	std::vector<QString> full;
	std::vector<QBrush> colors;
	tiny.resize(N);
	full.resize(N);
	colors.resize(N);
	for (int n=0; n<N; n++)
	{
		fprintf(stderr, "reading picks list: %d / %d\n", n+1, N);
		QFileInfo fileInfo = list[n];

	    QString filefullname = fileInfo.absoluteFilePath();
	    QString tinyName = fullFilenameToTinyName(filefullname);
	    tiny[n] = tinyName;
	    full[n] = filefullname;
	    colors[n] = QBrush(fullFilenameToColor(filefullname));
	    // qDebug() << tinyName;
	    // qDebug() << filefullname;
	}
	m_names.copy(tiny, full);
	m_allColors = colors;
}

void PicksManager::updateNamesFromDataBase(QString db_filename)
{
	fprintf(stderr, "--> %s\n", db_filename.toStdString().c_str());

	int N = 0;
	char buff[100000], buff2[10000], buff3[10000];
	int n0, t1, t2, t3;

	FILE *pFile = NULL;
	pFile = fopen(db_filename.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return;

	std::vector<QString> full;
	std::vector<QString> tiny;
	std::vector<QBrush> color;

	fscanf(pFile, "picks database\n", buff);
	fscanf(pFile, "number: %d\n", &N);
	tiny.resize(N);
	full.resize(N);
	color.resize(N);
	for (int n=0; n<N; n++)
	{
		fscanf(pFile, "%d %[^;];%[^;];%[^\n]\n", &n0, buff, buff2, buff3);
		tiny[n] = QString(buff);
		full[n] = QString(buff2);
		QColor _color = QColor(QString(buff3));
		color[n] = QBrush(_color);
	}
	m_names.copy(tiny, full);
	m_allColors = color;
}

void PicksManager::saveListToDataBase(QString db_filename)
{
	FILE *pFile = NULL;
	fprintf(stderr, "database filename: %s\n", db_filename.toStdString().c_str());
	pFile = fopen(db_filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	fprintf(pFile, "picks database\n");

	std::vector<QString> tiny = m_names.getTiny();
	std::vector<QString> full = m_names.getFull();
	const std::vector<QBrush>& colors = m_allColors;

	int N = tiny.size();
	fprintf(pFile, "number: %d\n", N);
	for (int n=0; n<N; n++)
	{
		fprintf(pFile, "%d %s;%s;%s\n", n, tiny[n].toStdString().c_str(), full[n].toStdString().c_str(),
				colors[n].color().name().toStdString().c_str());
   	}

	fclose(pFile);
	chmod(db_filename.toStdString().c_str(), (mode_t)0777);
}


void PicksManager::f_dataBaseUpdate()
{
	QString db_filename = getDatabaseName();
	updateNamesFromDisk();
	saveListToDataBase(db_filename);
	displayNames();
}


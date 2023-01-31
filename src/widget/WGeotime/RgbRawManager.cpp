#include <stdio.h>

#include <QDebug>
#include "Xt.h"
#include <iostream>
#include <sys/stat.h>
#include <RgbRawManager.h>

RgbRawManager::RgbRawManager(QWidget* parent) :
		ObjectManager(parent) {

}

RgbRawManager::~RgbRawManager()
{

}


void RgbRawManager::setSurvey(QString path, QString name)
{
	m_surveyPath = path;
	m_surveyName = name;
	m_RgbRawPath = m_surveyPath + RgbRawRootSubDir;
	// qDebug() << "[ SEISMIC ]" << m_seismicPath;
	updateNames();
	displayNames();
	// setSurveyName(name);
}

void RgbRawManager::setForceBasket(std::vector<QString> tinyName, std::vector<QString> fullName)
{
	m_namesBasket.copy(tinyName, fullName);
	displayNamesBasket();
}

std::vector<QString> RgbRawManager::getNames()
{
	return m_namesBasket.getTiny();
}

std::vector<QString> RgbRawManager::getPath()
{
	return m_namesBasket.getFull();
}

std::vector<QString> RgbRawManager::getAllDirectoryNames()
{
	// return directoryName;
	return m_names.getTiny();
}

std::vector<QString> RgbRawManager::getAllDirectoryPath()
{
	// return directoryPath;
	return m_names.getFull();
}

std::vector<QString> RgbRawManager::getAviNames()
{
	return m_aviNames.tiny;
}

std::vector<QString> RgbRawManager::getAviPath()
{
	return m_aviNames.full;
}



void RgbRawManager::updateNames()
{
	QString db_filename = getDatabaseName();
	// qDebug() << "[DATABASE] :" << db_filename;

	updateNamesFromDisk();
	updateDirectoryNames();
	return;
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

void RgbRawManager::updateNamesFromDisk()
{
	QDir *dir = new QDir(m_RgbRawPath);
	dir->setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	dir->setSorting(QDir::Name);
	QFileInfoList list = dir->entryInfoList();

	std::vector<QString> rgbFullName;
	std::vector<QString> rgbTinyName;
	int N = list.size();
	for (int i=0; i<N; i++)
	{
		QFileInfo fileInfo = list.at(i);
		QString path1 = m_RgbRawPath + fileInfo.fileName() + "/" + RgbRawRootSubDir2;
		QDir *dir0 = new QDir(path1);

		// dir0->setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
		// dir0->setSorting(QDir::Name);
		QStringList filters;
		filters << "*.raw" << "*.xt" << "*.rgb" << "*.avi";
		dir0->setNameFilters(filters);

		QFileInfoList list0 = dir0->entryInfoList();
		int NN = list0.size();
		for (int ii=0; ii<NN; ii++)
		{
			QFileInfo fileInfo0 = list0.at(ii);
			QString path0 = fileInfo0.fileName();
			QString tmp = path1 + path0;
			rgbFullName.push_back(tmp);
			QString rawname = path0.split(".",Qt::SkipEmptyParts).at(0);
			rgbTinyName.push_back(rawname);
		}
	}
	m_names.copy(rgbTinyName, rgbFullName);
	delete dir;
	// === avi
	std::vector<QString> aviName;
	std::vector<QString> aviPath;
	dir = new QDir(m_RgbRawPath);
	dir->setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	dir->setSorting(QDir::Name);
	QFileInfoList listAvi = dir->entryInfoList();
	for (int n=0; n<listAvi.size(); n++)
	{
		QString dirFilename = m_RgbRawPath + listAvi.at(n).fileName() + '/' + RgbRawRootSubDir2;
		QDir dir1(dirFilename);
		dir1.setFilter(QDir::Files);
		// dir->setFilter(QDir::AllEntries );
		dir1.setSorting(QDir::Name);
		QStringList filters;
		filters << "*.avi";
		dir1.setNameFilters(filters);
		QFileInfoList list0 = dir1.entryInfoList();
		int NN = list0.size();
		for (int ii=0; ii<NN; ii++)
		{
			QFileInfo fileInfo0 = list0.at(ii);
			QString path0 = fileInfo0.fileName();
			QString tmp = dirFilename + path0;
			aviPath.push_back(tmp);
			QString name = path0.split(".",Qt::SkipEmptyParts).at(0);
			aviName.push_back(name);
		}
	}
	m_aviNames.copy(aviName, aviPath);
}




/*
void RgbRawManager::updateNamesFromDisk()
{
	QDir dir(m_RgbRawPath);
	dir.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	dir.setSorting(QDir::Name);
	QFileInfoList list = dir.entryInfoList();

	std::vector<QString> rgbFullName;
	std::vector<QString> rgbTinyName;
	int N = list.size();
	for (int i=0; i<N; i++)
	{
		QFileInfo fileInfo = list.at(i);
		QString path1 = m_RgbRawPath + fileInfo.fileName() + "/" + RgbRawRootSubDir2;
		QDir dir0(path1);
		dir0.setFilter(QDir::Files);
		// dir->setFilter(QDir::AllEntries );
		dir0.setSorting(QDir::Name);
		QStringList filters;

		filters << "*.raw" << "*.xt";
		dir0.setNameFilters(filters);
		QFileInfoList list0 = dir0.entryInfoList();

		filters << "*.raw" << "*.xt" << "*.rgb" << "*.avi";
		dir0->setNameFilters(filters);
		QFileInfoList list0 = dir0->entryInfoList();

		int NN = list0.size();
		for (int ii=0; ii<NN; ii++)
		{
			QFileInfo fileInfo0 = list0.at(ii);
			QString path0 = fileInfo0.fileName();
			QString tmp = path1 + path0;
			rgbFullName.push_back(tmp);
			QString rawname = path0.split(".",Qt::SkipEmptyParts).at(0);
			rgbTinyName.push_back(rawname);
		}
	}
	m_names.copy(rgbTinyName, rgbFullName);
}
*/

void RgbRawManager::updateDirectoryNames()
{
	QDir dir(m_RgbRawPath);
	dir.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	dir.setSorting(QDir::Name);
	QFileInfoList list = dir.entryInfoList();

	directoryName.clear();
	directoryPath.clear();
	int N = list.size();
	for (int i=0; i<N; i++)
	{
		QFileInfo fileInfo = list.at(i);
		QString path1 = m_RgbRawPath + fileInfo.fileName() + "/" + RgbRawRootSubDir2;
		QDir dir0(path1);
		dir0.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
		// dir->setFilter(QDir::AllEntries );
		dir0.setSorting(QDir::Name);
		QFileInfoList list0 = dir0.entryInfoList();
		int NN = list0.size();
		for (int ii=0; ii<NN; ii++)
		{
			directoryName.push_back(list0[ii].fileName());
			directoryPath.push_back(list0[ii].absoluteFilePath());
		}
	}
}

void RgbRawManager::updateNamesFromDataBase(QString db_filename)
{

}



void RgbRawManager::saveListToDataBase(QString db_filename)
{

}

QString RgbRawManager::getDatabaseName()
{
	return "";
}


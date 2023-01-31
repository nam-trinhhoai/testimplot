
#include <stdio.h>
#include <QDebug>
#include <QDir>
#include <freeHorizonManager.h>
#include <workingsetmanager.h>
#include <GeotimeProjectManagerWidget.h>
#include <ProjectManagerWidget.h>
#include <QFileInfoList>
#include <QFileInfo>
#include <QStringList>
#include <freeHorizonManager.h>
#include <Xt.h>

#include <nvhorizonconvertion.h>

bool NVHorizonConvertion::convertion(GeotimeProjectManagerWidget* manager)
{
	if ( manager == nullptr ) return false;
	QString path = manager->get_IJKPath() + "HORIZONS/";
	qDebug() << path;

	QString oldHorizonBaseDir = QString::fromStdString(FreeHorizonManager::OldBaseDirectory);
	// QString newHorizonBaseDir = QString::fromStdString(FreeHorizonManager::BaseDirectory);

	QString oldHorizonDir = path + oldHorizonBaseDir;
	QString newHorizonDir = manager->getNVHorizonPath();

	QDir oldDir(oldHorizonDir);
	if ( !oldDir.exists() ) return true;
	qDebug() << "enter NV-HORIZON convertion";
	return convertion(oldHorizonDir, newHorizonDir);
}

bool NVHorizonConvertion::convertion(ProjectManagerWidget* manager)
{
	if ( manager == nullptr ) return false;
	QString path = manager->getIJKPath() + "HORIZONS/";
	qDebug() << path;

	QString oldHorizonBaseDir = QString::fromStdString(FreeHorizonManager::OldBaseDirectory);
	// QString newHorizonBaseDir = QString::fromStdString(FreeHorizonManager::BaseDirectory);

	QString oldHorizonDir = path + oldHorizonBaseDir;
	QString newHorizonDir = manager->getNVHorizonPath();

	QDir oldDir(oldHorizonDir);
	if ( !oldDir.exists() ) return true;
	qDebug() << "enter NV-HORIZON convertion";
	return convertion(oldHorizonDir, newHorizonDir);
}


bool NVHorizonConvertion::convertion(QString oldPath, QString newPath)
{
	std::vector<QString> out;
	QFileInfoList infoList = QDir(oldPath).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
	{
		qDebug() << infoList[i].absoluteFilePath();
		qDebug() << infoList[i].baseName();
		directoryConvertion(oldPath, infoList[i].baseName(), newPath);
	}
	// QDir d;
	// bool ret = d.rmpath(oldPath);
	QDir dir(oldPath);
	int ret = dir.removeRecursively();
	return true;
}

QString NVHorizonConvertion::newNameGet(QString oldName)
{
	QStringList list = oldName.split("_(");
	if ( list.size() > 0 )
		return list[0];
	return "";
}

bool NVHorizonConvertion::directoryConvertion(QString oldBasePath, QString oldName, QString newBasePath)
{
	QString newName = newNameGet(oldName);
	QString newPath = newBasePath + "/" + newName;
	QDir dir(newPath);
	if ( !dir.exists() )
	{
		QDir d;
		if ( !d.mkpath(newPath) )
		{
			qDebug() << "unable to create directory: " << newPath;
		}
		directoryFileConvertion(oldBasePath+"/"+oldName, newPath);
	}
	return true;
}


std::vector<QString> NVHorizonConvertion::getFiles(QString path)
{
	std::vector<QString> names;

	QDir dir(path);
    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);
    QStringList filters;
    filters << "*.*";
    dir.setNameFilters(filters);
    QFileInfoList list = dir.entryInfoList();
    int N = list.size();
    names.resize(N);
    for (int i=0; i<list.size(); i++)
    {
        QFileInfo fileInfo = list[i];
        names[i] = fileInfo.fileName();
    }
    return names;
}

int NVHorizonConvertion::getDimvfromRaw(QString filename, int dimy, int dimz)
{
	FILE *pFile = fopen((char*)filename.toStdString().c_str() , "r");
	if ( !pFile ) return 0;
	fseek(pFile, 0, SEEK_END);
	long fileSize = (unsigned long)ftell(pFile);
	int nfreq = (int)(fileSize / (dimy*dimz*sizeof(short)));
	fclose(pFile);
	return nfreq;
}

bool NVHorizonConvertion::attributConversion(QString iso, QString src, QString dst)
{
	inri::Xt xtRef(iso.toStdString().c_str());
	if ( !xtRef.is_valid() ) return false;
	int dimy = xtRef.nSamples();
	int dimz = xtRef.nRecords();
	int dimv = getDimvfromRaw(src, dimy, dimz);
	short *tmp = (short*)calloc((long)dimy*dimz*dimv, sizeof(short));
	if ( tmp == nullptr) return false;
	FILE *pf = fopen(src.toStdString().c_str(), "r");
	if ( pf == nullptr ) return false;
	fread(tmp, sizeof(short), (long)dimy*dimz*dimv, pf);
	fclose(pf);
	FreeHorizonManager::attributWriteRefAttribut(iso.toStdString(), dst.toStdString(), tmp, dimv);
	QFileInfo fi(dst);
	QString name = fi.baseName();
	std::string type = FreeHorizonManager::typeFromAttributName(name.toStdString());
	if ( type == "spectrum")
	{
		int fgreen = FreeHorizonManager::spectrumOptimalFGreenIndexProcess(dst.toStdString());
	}
	free(tmp);
	return true;
}

bool NVHorizonConvertion::directoryFileConvertion(QString oldPath, QString newPath)
{
	std::vector<QString> names = getFiles(oldPath);
	// for (QString str:names) qDebug() << str;
	for (int i=0; i<names.size(); i++)
	{
		QFileInfo fi(oldPath+"/"+names[i]);
		if ( fi.suffix() == "txt" || fi.suffix() == "iso" )
		{
			QFile::copy(oldPath+"/"+names[i], newPath+"/"+names[i]);
		}
		else if ( fi.suffix() == "raw" )
		{
			attributConversion(oldPath+"/"+"isochrone.iso", oldPath+"/"+names[i], newPath+"/"+fi.baseName()+".amp");
		}
	}
	return true;
}

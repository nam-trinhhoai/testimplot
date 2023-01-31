
#ifndef __SEISMICDATABASEMANAGER__
#define __SEISMICDATABASEMANAGER__


#include <QString>
#include <QBrush>
#include <vector>

class SeismisDatabaseManager
{
public:
	class Data
	{
	public:
		QString path = "";
		QString name = "";
		QString fullname = "";
		QBrush color = Qt::white;
		int dimx = 0;
		int dimy = 0;
		int dimz = 0;
	};
public:
	static std::vector<Data> update(QString databasePath, QString datasetPath);
	static std::vector<Data> databaseRead(QString databasePath);
	static std::vector<QString> seismicDiskRead(QString path);
	static bool isDataInDatabase(QString &seismicDiskPath, std::vector<SeismisDatabaseManager::Data> &databaseData);
	static bool isDataInDisk(QString &seismicPath, std::vector<QString> &seismicDiskPath);
	static int getIndexFromVector(QString &path, std::vector<SeismisDatabaseManager::Data> &databaseData);
	static int filextGetAxis(QString filename);
	static SeismisDatabaseManager::Data seismciGetDataFromDisk(QString seismicPath);
	static QString seismicGetNameFromPath(QString filename);
	static void databaseWrite(QString filename, std::vector<SeismisDatabaseManager::Data> &vData);
private:
	static std::vector<QString> seismicLineSplit(QString dataIn);
	static int dataBaseVersion(QString databasePath);
};




#endif

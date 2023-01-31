
#ifndef __WELLSDATABASEMANAGER__
#define __WELLSDATABASEMANAGER__

#include <QString>

#include <vector>

#include <boost/filesystem.hpp>
#include <QFileInfoList>
#include <WellUtil.h>


class WellsDatabaseManager
{
public:
	// static QString deviationFilename;
	static void update(QString databasePath, QString datasetPath);



private:
	static std::vector<WELLHEADDATA> databaseRead(QString databasePath);
	static void databaseWrite(QString filename, std::vector<WELLHEADDATA>& data);
	static QString getDeviationNames(QString path);
	static std::vector<WELLHEADDATA> wellDiskRead(QString path);
	static QFileInfoList getDir(QString path);
	static std::vector<QString> getLogFilenames(QString path);
	static QFileInfoList getFiles(std::string path, QStringList ext);
	static std::vector<QString> getTF2PFilenames(QString path);
	static std::vector<QString> getPicksFilenames(QString path);
	static bool isWellHeadInDatabase(WELLHEADDATA &data, std::vector<WELLHEADDATA> &databaseData);
	static WELLHEADDATA wellHeadCreate(QString path);
	static WELLBOREDATA wellBoreCreate(QString headName, QString path);

	static std::pair<std::vector<QString>, std::vector<QString>> logListCreate(std::string path0);
	static std::pair<std::vector<QString>, std::vector<QString>> tf2pListCreate(std::string path0);
	static std::pair<std::vector<QString>, std::vector<QString>> picksListCreate(std::string path0);
	static int getIndexFromVector(WELLHEADDATA &path, std::vector<WELLHEADDATA> &databaseData);

	static int getIdxDatabase(std::vector<WELLHEADDATA> &dataBase, QString path);
	static int getIdxBore(WELLHEADDATA &data, QString path);

	static bool wellHeadUpdate(std::vector<WELLHEADDATA> &databaseData, std::vector<WELLHEADDATA> &diskData);
	static void wellLogUpdate(std::vector<WELLHEADDATA> &databaseData, std::vector<WELLHEADDATA> &diskData);
	static void wellBoreUpdate(std::vector<WELLHEADDATA> &databaseData, std::vector<WELLHEADDATA> &diskData);

	static bool isWellHeadFit(WELLHEADDATA head0, std::vector<WELLHEADDATA> &base);
	static bool isWellHeadExists(WELLHEADDATA head0, std::vector<WELLHEADDATA> &base);
	static bool isWellBoreFit(std::vector<WELLBOREDATA> &base1, std::vector<WELLBOREDATA> &base2);




};



static const QString deviationFilename = QStringLiteral("deviation");










#endif

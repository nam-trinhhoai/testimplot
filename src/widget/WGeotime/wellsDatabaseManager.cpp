
#include <stdio.h>

#include <QFile>

#include <sys/stat.h>
#include <QDir>
#include <QFileInfoList>
#include <ProjectManagerNames.h>
#include "utils/stringutil.h"
#include <wellsDatabaseManager.h>

namespace fs = boost::filesystem;


void  WellsDatabaseManager::update(QString databasePath, QString datasetPath)
{
	std::vector<WELLHEADDATA> databaseData;
	std::vector<WELLHEADDATA> diskData;
	if ( databasePath.isEmpty() || datasetPath.isEmpty() ) return;

	databaseData = databaseRead(databasePath);
	diskData = wellDiskRead(datasetPath);

	bool update = wellHeadUpdate(databaseData, diskData);

	if ( update )
		databaseWrite(databasePath, databaseData);
	else
		fprintf(stderr, "the database is unchanged\n");
}

bool WellsDatabaseManager::wellHeadUpdate(std::vector<WELLHEADDATA> &databaseData, std::vector<WELLHEADDATA> &diskData)
{
	std::vector<QString> remove;
	for (int n=0; n<databaseData.size(); n++)
	{
		bool ok = isWellHeadFit(databaseData[n], diskData);
		if ( !ok )
			remove.push_back(databaseData[n].fullName);
	}

	for (QString path:remove)
	{
		int idx = getIdxDatabase(databaseData, path);
		if ( idx < 0 ) continue;
		databaseData.erase(databaseData.begin()+idx, databaseData.begin()+idx+1);
	}

	std::vector<QString> add;
	for (int n=0; n<diskData.size(); n++)
	{
		bool ok = isWellHeadExists(diskData[n], databaseData);
		if ( !ok )
			add.push_back(diskData[n].fullName);
	}
	for (QString path:add)
	{
		databaseData.push_back(wellHeadCreate(path));
	}

	if ( remove.size() > 0 ) fprintf(stderr, "well database remove: %d\n", remove.size());
	if ( add.size() > 0 ) fprintf(stderr, "welld atabase add: %d\n", add.size());

	if ( remove.size() > 0 || add.size() > 0 ) return true;
	return false;
}



bool WellsDatabaseManager::isWellHeadFit(WELLHEADDATA head0, std::vector<WELLHEADDATA> &base)
{
	for (int n=0; n<base.size(); n++)
	{
		if ( base[n].fullName == head0.fullName )
		{
			return isWellBoreFit(base[n].bore, head0.bore);
		}
	}
	return false;
}


bool WellsDatabaseManager::isWellBoreFit(std::vector<WELLBOREDATA> &base1, std::vector<WELLBOREDATA> &base2)
{
	if ( base1.size() != base2.size() ) return false;
	for (int n=0; n<base1.size(); n++)
	{
		if ( base1[n].fullName != base2[n].fullName ) return false;

		if ( base1[n].logs.full.size() != base2[n].logs.full.size() ) return false;
		for (int i=0; i<base1[n].logs.full.size(); i++)
			if ( base1[n].logs.full[i] != base2[n].logs.full[i] ) return false;

		if ( base1[n].tf2p.full.size() != base2[n].tf2p.full.size() ) return false;
		for (int i=0; i<base1[n].tf2p.full.size(); i++)
			if ( base1[n].tf2p.full[i] != base2[n].tf2p.full[i] ) return false;

		if ( base1[n].picks.full.size() != base2[n].picks.full.size() ) return false;
		for (int i=0; i<base1[n].picks.full.size(); i++)
			if ( base1[n].picks.full[i] != base2[n].picks.full[i] ) return false;
	}
	return true;
}

bool WellsDatabaseManager::isWellHeadExists(WELLHEADDATA head0, std::vector<WELLHEADDATA> &base)
{
	for (WELLHEADDATA well:base)
	{
		if ( well.fullName == head0.fullName ) return true;
	}
	return false;
}




void WellsDatabaseManager::wellBoreUpdate(std::vector<WELLHEADDATA> &databaseData, std::vector<WELLHEADDATA> &diskData)
{
	/*
	for (int n=0; n<databaseData.size(); n++)
	{
		int idxDisk = getIdxDatabase(diskData, databaseData[n].fullName);
		for (int i=0; i<databaseData[n].bore.size(); i++)
		{
			bool same = isBoreSame(databaseData[n].bore[i], diskData[idxDisk].bore);
			if ( !same )
			{
				// boreSuppress(databaseData[n], )
			}
		}
	}
	*/
}

int WellsDatabaseManager::getIdxDatabase(std::vector<WELLHEADDATA> &dataBase, QString path)
{
	for (int n=0; n<dataBase.size(); n++)
		if ( dataBase[n].fullName == path ) return n;
	return -1;
}

int WellsDatabaseManager::getIdxBore(WELLHEADDATA &data, QString path)
{
	for (int i=0; i<data.bore.size(); i++)
		if ( data.bore[i].fullName == path ) return i;
	return -1;
}

void WellsDatabaseManager::wellLogUpdate(std::vector<WELLHEADDATA> &databaseData, std::vector<WELLHEADDATA> &diskData)
{
	for (int n=0; n<databaseData.size(); n++)
	{
		int idxDisk = getIdxDatabase(diskData, databaseData[n].fullName);
		if ( idxDisk < 0 ) continue;
		for (int i=0; i<databaseData[n].bore.size(); i++)
		{
			int idxBore = getIdxBore(diskData[idxDisk], databaseData[n].fullName);


			std::vector<QString> newLog;
			// for (int j=0; j<)


		}
	}
}

bool WellsDatabaseManager::isWellHeadInDatabase(WELLHEADDATA &data, std::vector<WELLHEADDATA> &databaseData)
{
	for (int i=0; i<databaseData.size(); i++)
	{
		if ( data.fullName == databaseData[i].fullName ) return true;
	}
	return false;
}


std::vector<WELLHEADDATA> WellsDatabaseManager::wellDiskRead(QString path)
{
	std::vector<WELLHEADDATA> data;

	QFileInfoList list = getDir(path);
	int N = list.size();
	data.resize(N);

	for (int n_well=0; n_well<N; n_well++)
	{
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

		data[n_well].tinyName = filetinyname;
		data[n_well].fullName = filefullname;

		QFileInfoList list_bore = getDir(filefullname);
		int Nbores = list_bore.size();
		data[n_well].bore.resize(Nbores);

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
			data[n_well].bore[n_bore].tinyName = QString("[ ") + filetinyname +QString(" ] ") + bore_filetinyname;
			data[n_well].bore[n_bore].fullName = bore_filefullname;

			std::vector<QString> logFilenames = getLogFilenames(bore_filefullname);
			std::vector<QString> tf2pFilenames = getTF2PFilenames(bore_filefullname);
			std::vector<QString> picksFilenames = getPicksFilenames(bore_filefullname);

			data[n_well].bore[n_bore].logs.full.resize(logFilenames.size());
			data[n_well].bore[n_bore].logs.tiny.resize(logFilenames.size(), "idle");
			for (int i=0; i<logFilenames.size(); i++)
				data[n_well].bore[n_bore].logs.full[i] = logFilenames[i];

			data[n_well].bore[n_bore].tf2p.full.resize(tf2pFilenames.size());
			data[n_well].bore[n_bore].tf2p.tiny.resize(tf2pFilenames.size(), "idle");
			for (int i=0; i<tf2pFilenames.size(); i++)
				data[n_well].bore[n_bore].tf2p.full[i] = tf2pFilenames[i];

			data[n_well].bore[n_bore].picks.full.resize(picksFilenames.size());
			data[n_well].bore[n_bore].picks.tiny.resize(picksFilenames.size(), "idle");
			for (int i=0; i<picksFilenames.size(); i++)
				data[n_well].bore[n_bore].picks.full[i] = picksFilenames[i];
		}
//		fprintf(stderr, "%d %d\n", n_well, N);
	}
	return data;
}




std::vector<WELLHEADDATA> WellsDatabaseManager::databaseRead(QString databasePath)
{
	// fprintf(stderr, "--> %s\n", databasePath.toStdString().c_str());

	std::vector<WELLHEADDATA> data0;
	int nwells = 0;
	char buff[100000], buff2[10000];
	FILE *pFile = nullptr;
	pFile = fopen((char*)databasePath.toStdString().c_str(), "rb");
	if ( pFile == nullptr ) return data0;

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
	return data0;
}

void WellsDatabaseManager::databaseWrite(QString filename, std::vector<WELLHEADDATA> &data)
{
	FILE *pFile = NULL;
	// fprintf(stderr, "database filename: %s\n", filename.toStdString().c_str());
	pFile = fopen(filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	fprintf(pFile, "Wells database\n");
	fprintf(pFile, "Wells number: %d\n", data.size());
	for (int n=0; n<data.size(); n++)
	{
	    fprintf(pFile, "head:%d %s;%s\n", n, data[n].tinyName.toStdString().c_str(), data[n].fullName.toStdString().c_str());
	    int Nbore = data[n].bore.size();
	    fprintf(pFile, "bore number: %d\n", Nbore);
	    for (int n2=0; n2<Nbore; n2++)
	    {
	    	fprintf(pFile, "head: %d bore: %d %s;%s\n", n, n2, data[n].bore[n2].tinyName.toStdString().c_str(), data[n].bore[n2].fullName.toStdString().c_str());
	    	int N2 = data[n].bore[n2].logs.tiny.size();
	    	fprintf(pFile, "logs number: %d\n", N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fprintf(pFile, "head: %d bore: %d log: %d %s;%s\n", n, n2, n3, data[n].bore[n2].logs.tiny[n3].toStdString().c_str(), data[n].bore[n2].logs.full[n3].toStdString().c_str());
	    	}
	    	N2 = data[n].bore[n2].tf2p.tiny.size();
	    	fprintf(pFile, "tf2p number: %d\n", N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fprintf(pFile, "head: %d bore: %d tf2p: %d %s;%s\n", n, n2, n3, data[n].bore[n2].tf2p.tiny[n3].toStdString().c_str(), data[n].bore[n2].tf2p.full[n3].toStdString().c_str());
	    	}
	    	N2 = data[n].bore[n2].picks.tiny.size();
	    	fprintf(pFile, "picks number: %d\n", N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fprintf(pFile, "head: %d bore: %d picks: %d %s;%s\n", n, n2, n3, data[n].bore[n2].picks.tiny[n3].toStdString().c_str(), data[n].bore[n2].picks.full[n3].toStdString().c_str());
	    	}
	    }
	}
	fclose(pFile);
	chmod(filename.toStdString().c_str(), (mode_t)0777);
}


QString WellsDatabaseManager::getDeviationNames(QString path)
{
	QString filename = path + "/" + deviationFilename;
	if ( QFile::exists(filename) )
	{
		return filename;
	}
	return "";
}

QFileInfoList WellsDatabaseManager::getDir(QString path)
{
	QDir dir(path);
	dir.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	dir.setSorting(QDir::Name);
	QFileInfoList list = dir.entryInfoList();
	return list;
}


std::vector<QString> WellsDatabaseManager::getLogFilenames(QString path)
{

	QStringList filters;
	filters << "*.log";
	QFileInfoList list = getFiles(path.toStdString(), filters);
	std::vector<QString> data;
	data.resize(list.size());
	for (int i=0; i<list.size(); i++)
		data[i] = list[i].absoluteFilePath();
	return data;
}

std::vector<QString> WellsDatabaseManager::getTF2PFilenames(QString path)
{

	QStringList filters;
	filters << "*.tfp";
	QFileInfoList list = getFiles(path.toStdString(), filters);
	std::vector<QString> data;
	data.resize(list.size());
	for (int i=0; i<list.size(); i++)
		data[i] = list[i].absoluteFilePath();
	return data;
}

std::vector<QString> WellsDatabaseManager::getPicksFilenames(QString path)
{

	QStringList filters;
	filters << "*.pick";
	QFileInfoList list = getFiles(path.toStdString(), filters);
	std::vector<QString> data;
	data.resize(list.size());
	for (int i=0; i<list.size(); i++)
		data[i] = list[i].absoluteFilePath();
	return data;
}

QFileInfoList WellsDatabaseManager::getFiles(std::string path, QStringList ext)
{
	QFileInfoList list;

	if(fs::exists(path)) {
		for( const auto & entry : fs::directory_iterator(path)) {
			for(int i=0;i<ext.size();i++) {
				QString newext = ext[i].replace("*","");
				if ( endsWith(entry.path().c_str(), newext.toStdString()) == 1) {
					if(fs::is_regular_file(entry.path().c_str()) ) {
						list.append(QFileInfo(QString::fromStdString(entry.path().c_str())));
					}
				}
			}
		}
	}
	return list;
}



// ======================= create


WELLHEADDATA WellsDatabaseManager::wellHeadCreate(QString path)
{
	WELLHEADDATA data;
	QDir dir(path);
	QString dirName = dir.dirName();
	QString descFile = path + "/" + dirName + ".desc";
	data.fullName = path;
	data.tinyName = ProjectManagerNames::getKeyTabFromFilename(descFile, "Name");
	if ( data.tinyName.isNull() || data.tinyName.isEmpty() ) data.tinyName = dirName;

	QFileInfoList list = getDir(path);
	data.bore.resize(list.size());
	for (int i=0; i<list.size(); i++)
		data.bore[i] = wellBoreCreate(data.tinyName, list[i].absoluteFilePath());
	return data;
}


WELLBOREDATA WellsDatabaseManager::wellBoreCreate(QString headName, QString path)
{
	WELLBOREDATA data;
	QDir dir(path);
	QString dirName = dir.dirName();
	QString descFile = path + "/" + dirName + ".desc";
	data.fullName = path;
	data.tinyName = "[ " + headName + " ] " + ProjectManagerNames::getKeyTabFromFilename(descFile, "Name");

	std::pair<std::vector<QString>, std::vector<QString>> logs = logListCreate(path.toStdString());
	data.logs.full.resize(logs.second.size());
	data.logs.tiny.resize(logs.first.size());
	for (int i=0; i<logs.second.size(); i++)
	{
		data.logs.full[i] = logs.second[i];
		data.logs.tiny[i] = logs.first[i];
	}

	std::pair<std::vector<QString>, std::vector<QString>> tf2p = tf2pListCreate(path.toStdString());
	data.tf2p.full.resize(tf2p.second.size());
	data.tf2p.tiny.resize(tf2p.first.size());
	for (int i=0; i<tf2p.second.size(); i++)
	{
		data.tf2p.full[i] = tf2p.second[i];
		data.tf2p.tiny[i] = tf2p.first[i];
	}

	std::pair<std::vector<QString>, std::vector<QString>> picks = picksListCreate(path.toStdString());
	data.picks.full.resize(picks.second.size());
	data.picks.tiny.resize(picks.first.size());
	for (int i=0; i<picks.second.size(); i++)
	{
		data.picks.full[i] = picks.second[i];
		data.picks.tiny[i] = picks.first[i];
	}
	return data;
}


std::pair<std::vector<QString>, std::vector<QString>> WellsDatabaseManager::logListCreate(std::string path0)
{
	std::vector<QString> name;
	std::vector<QString> path;

	QStringList filters;
	filters << "*.log";
	QFileInfoList list = getFiles(path0, filters);
	path.resize(list.size());
	name.resize(list.size());
	for (int i=0; i<list.size(); i++)
	{
		QFileInfo f = list[i];
		path[i] = f.absoluteFilePath();
		name[i] = ProjectManagerNames::getKeyTabFromFilename(path[i], "Name");
	}
	return std::make_pair(name, path);
}

std::pair<std::vector<QString>, std::vector<QString>> WellsDatabaseManager::tf2pListCreate(std::string path0)
{
	std::vector<QString> name;
	std::vector<QString> path;

	QStringList filters;
	filters << "*.tfp";
	QFileInfoList list = getFiles(path0, filters);
	path.resize(list.size());
	name.resize(list.size());
	for (int i=0; i<list.size(); i++)
	{
		QFileInfo f = list[i];
		path[i] = f.absoluteFilePath();
		name[i] = ProjectManagerNames::getKeyTabFromFilename(path[i], "Name");
	}
	return std::make_pair(name, path);
}

std::pair<std::vector<QString>, std::vector<QString>> WellsDatabaseManager::picksListCreate(std::string path0)
{
	std::vector<QString> name;
	std::vector<QString> path;

	QStringList filters;
	filters << "*.pick";
	QFileInfoList list = getFiles(path0, filters);
	path.resize(list.size());
	name.resize(list.size());
	for (int i=0; i<list.size(); i++)
	{
		QFileInfo f = list[i];
		path[i] = f.absoluteFilePath();
		name[i] = ProjectManagerNames::getKeyTabFromFilename(path[i], "Name");
	}
	return std::make_pair(name, path);
}


int WellsDatabaseManager::getIndexFromVector(WELLHEADDATA &path, std::vector<WELLHEADDATA> &databaseData)
{
	for ( int n=0; n<databaseData.size(); n++ )
	{
		if ( path.fullName == databaseData[n].fullName ) return n;
	}
	return -1;
}

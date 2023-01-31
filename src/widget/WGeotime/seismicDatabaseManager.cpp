
#include <stdio.h>
#include <QProcess>
#include <iostream>
#include <sys/stat.h>

#include <geotimepath.h>
#include <Xt.h>
#include <SeismicManager.h>
#include <seismicDatabaseManager.h>



std::vector<SeismisDatabaseManager::Data> SeismisDatabaseManager::update(QString databasePath, QString datasetPath)
{
	std::vector<SeismisDatabaseManager::Data> data;
	if ( databasePath.isEmpty() || datasetPath.isEmpty() ) return data;
	int version = dataBaseVersion(databasePath);

	std::vector<SeismisDatabaseManager::Data> databaseData;
	if ( version == 2 )
	{
		databaseData = databaseRead(databasePath);
	}
	std::vector<QString> seismicDiskPath = seismicDiskRead(datasetPath);

	int NseismicDisk = seismicDiskPath.size();
	std::vector<QString> newPath;
	for (int n=0; n<NseismicDisk; n++)
	{
		if ( !isDataInDatabase(seismicDiskPath[n], databaseData) )
		{
			newPath.push_back(seismicDiskPath[n]);
		}
	}
	std::vector<QString> oldPath;
	for (int n=0; n<databaseData.size(); n++)
	{
		if ( !isDataInDisk(databaseData[n].path, seismicDiskPath) )
		{
			oldPath.push_back(databaseData[n].path);
		}
	}

	for ( QString path:oldPath)
	{
		int idx = getIndexFromVector(path, databaseData);
		if ( idx >= 0 )
			databaseData.erase(databaseData.begin()+idx, databaseData.begin()+idx+1);
	}

	for ( QString path:newPath )
	{
		SeismisDatabaseManager::Data data = seismciGetDataFromDisk(path);
		databaseData.push_back(data);
	}

	if ( oldPath.size() != 0 || newPath.size() != 0 )
		databaseWrite(databasePath, databaseData);

	if ( oldPath.size() > 0 )
	{
		qDebug() << QString::number(oldPath.size()) << " data is (are) removed from the database";
	}
	if ( newPath.size() > 0 )
	{
		qDebug() << QString::number(newPath.size()) << " data is (are) added to the database";
	}
	if ( oldPath.size() != 0 || newPath.size() != 0 )
		qDebug() << "the database was updated";
	else
		qDebug() << "the database is unchanged";
	return databaseData;
}


int SeismisDatabaseManager::getIndexFromVector(QString &path, std::vector<SeismisDatabaseManager::Data> &databaseData)
{
	for ( int n=0; n<databaseData.size(); n++ )
	{
		if ( path == databaseData[n].path ) return n;
	}
	return -1;
}

bool SeismisDatabaseManager::isDataInDatabase(QString &seismicDiskPath, std::vector<SeismisDatabaseManager::Data> &databaseData)
{
	for (SeismisDatabaseManager::Data data:databaseData)
	{
		if ( seismicDiskPath == data.path ) return true;
	}
	return false;
}

bool SeismisDatabaseManager::isDataInDisk(QString &seismicPath, std::vector<QString> &seismicDiskPath)
{
	for (QString data:seismicDiskPath)
	{
		if ( seismicPath == data ) return true;
	}
	return false;
}


std::vector<QString> SeismisDatabaseManager::seismicLineSplit(QString dataIn)
{
	std::vector<QString> str;
	QStringList out0 = dataIn.split(" ");
	if ( out0.size() < 2 ) return str;
	str.push_back(out0[0]);
	QStringList cutList(QList<QString>(out0.begin()+1, out0.end()));
	QString troncatedData = cutList.join(" ");
	QStringList out1 = troncatedData.split(";");
	for (int i=0; i<out1.size(); i++)
	{
		out1[i].replace("\n", "");
		str.push_back(out1[i]);
	}
	return str;
}

std::vector<SeismisDatabaseManager::Data> SeismisDatabaseManager::databaseRead(QString databasePath)
{
	std::vector<SeismisDatabaseManager::Data> data;
	if ( databasePath.isEmpty() ) return data;

	int N = 0, n0 = 0;
	char buff[100000], buff2[10000], buff3[100000], buff4[100000], buff5[100000], buff6[100000];
	int dimx0, dimy0, dimz0;
	FILE *pFile = NULL;
	pFile = fopen(databasePath.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return data;

	fscanf(pFile, "Seismic database\n", buff);
	fscanf(pFile, "Seismic number: %d\n", &N);
	data.resize(N);
	for (int n=0; n<N; n++)
	{
		char *res0 = fgets(buff, 100000, pFile);
		QString qres0(res0);
		std::vector<QString> str = seismicLineSplit(qres0);
		if ( str.size() < 4 ) continue;
		data[n].name = str[1];
		data[n].path = str[2];
		QFileInfo fi(data[n].path);
		data[n].fullname = fi.fileName();
		QBrush brush = Qt::white;
		if ( str[3] == SeismicManager::TIME_SHORT_COLOR_STR ) brush = SeismicManager::TIME_SHORT_COLOR;
		if ( str[3] == SeismicManager::TIME_8BIT_COLOR_STR ) brush = SeismicManager::TIME_8BIT_COLOR;
		if ( str[3] == SeismicManager::TIME_32BIT_COLOR_STR ) brush = SeismicManager::TIME_32BIT_COLOR;
		if ( str[3] == SeismicManager::DEPTH_SHORT_COLOR_STR ) brush = SeismicManager::DEPTH_SHORT_COLOR;
		if ( str[3] == SeismicManager::DEPTH_8BIT_COLOR_STR ) brush = SeismicManager::DEPTH_8BIT_COLOR;
		if ( str[3] == SeismicManager::DEPTH_32BIT_COLOR_STR ) brush = SeismicManager::DEPTH_32BIT_COLOR;
		if ( str[3] == "green" ) brush = Qt::green;
		data[n].color = brush;
		if ( str.size() == 7 )
		{
			data[n].dimx = str[4].toInt();
			data[n].dimy = str[5].toInt();
			data[n].dimz = str[6].toInt();
		}
	}
	fclose(pFile);
	return data;
}


std::vector<QString> SeismisDatabaseManager::seismicDiskRead(QString path)
{
	std::vector<QString> fullName;
	std::vector<QString> tinyName;

	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.xt" << "*.cwt";
	dir.setNameFilters(filters);
	QFileInfoList list = dir.entryInfoList();

	dir.cdUp();
	dir.cdUp();
	QString patchPath = dir.path() + "/" +  QString::fromStdString(GeotimePath::NEXTVISION_IMPORT_EXPORT_DIR) +
			"/" + QString::fromStdString(GeotimePath::NEXTVISION_MAIN_DIR) + "/" +
			QString::fromStdString(GeotimePath::NEXTVISION_SEISMIC_DIR);
	QDir dirPatch(patchPath);
	dirPatch.setFilter(QDir::Files);
	dirPatch.setSorting(QDir::Name);
	dirPatch.setNameFilters(filters);
	QFileInfoList listPatch = dirPatch.entryInfoList();

	int N = list.size() + listPatch.size();
	fullName.resize(N);
	tinyName.resize(N);
	int cpt = 0;
	for (int i=0; i<list.size(); i++)
	{
		QFileInfo fileInfo = list[i];
		QString filename = fileInfo.fileName();
		fullName[cpt++] = fileInfo.absoluteFilePath();
	}

	for (int i=0; i<listPatch.size(); i++)
	{
		QFileInfo fileInfo = listPatch[i];
		QString filename = fileInfo.fileName();
		fullName[cpt++] = fileInfo.absoluteFilePath();
	}
	return fullName;
}




SeismisDatabaseManager::Data SeismisDatabaseManager::seismciGetDataFromDisk(QString seismicPath)
{
	SeismisDatabaseManager::Data data;

	bool isValid = true;
	QBrush color0;
	QString name0 = seismicGetNameFromPath(seismicPath);
	QFileInfo fi(seismicPath);
	QString ext = fi.suffix();
	int dimx = 0;
	int dimy = 0;
	int dimz = 0;

	if ( ext.compare("xt") == 0 )
	{
		int axis = filextGetAxis(seismicPath);
		inri::Xt xt(seismicPath.toStdString().c_str());
		dimx = xt.nSamples();
		dimy = xt.nRecords();
		dimz = xt.nSlices();
		inri::Xt::Type type = inri::Xt::Unknown;
		if (axis==0 || axis==1) {
			if (xt.is_valid()) {
				type = xt.type();
			}
		}
		if (axis == 0 ) {
			switch (type) {
			case inri::Xt::Signed_16:
				color0 = SeismicManager::TIME_SHORT_COLOR;
				break;
			case inri::Xt::Unsigned_16:
			case inri::Xt::Unsigned_8:
			case inri::Xt::Signed_8:
				color0 = SeismicManager::TIME_8BIT_COLOR;
				break;
			default:
				color0 = SeismicManager::TIME_32BIT_COLOR;
				break;
			}
		} else if (axis == 1 ) {
			switch (type) {
			case inri::Xt::Signed_16:
				color0 = SeismicManager::DEPTH_SHORT_COLOR;
				break;
			case inri::Xt::Unsigned_16:
			case inri::Xt::Unsigned_8:
			case inri::Xt::Signed_8:
				color0 = SeismicManager::DEPTH_8BIT_COLOR;
				break;
			default:
				color0 = SeismicManager::DEPTH_32BIT_COLOR;
				break;
			}
		}
		else {
			color0 = Qt::white;
			isValid = false;
		}
	}
	else if ( ext.compare("cwt") == 0 )
	{
		name0 += " (compress)";
		color0 = Qt::green;
	}
	data.name = name0;
	data.color = color0;
	data.path = seismicPath;
	data.dimx = dimx;
	data.dimy = dimy;
	data.dimz = dimz;
	return data;
}


int SeismisDatabaseManager::filextGetAxis(QString filename)
{
	QProcess process;
	QStringList options;
	options << filename;
	process.start("TestXtFile", options);
	process.waitForFinished();

	if ( process.exitCode() != QProcess::NormalExit ) {
		std::cerr << "provided file is not in xt format (" << filename.toStdString() << ")" << std::endl;
		return 2;
	}

	// fprintf(stderr, "-> %s\n", filename.toStdString().c_str());
	// qDebug() << filename;
	std::size_t offset;
	{
		inri::Xt xt(filename.toStdString().c_str());
		if (!xt.is_valid()) {
			fprintf(stderr, "xt cube is not valid ( %s )\n", filename.toStdString().c_str());
			return 2;
		}
		offset = (size_t)xt.header_size();
	}
	QFile file(filename);
	size_t size = file.size();
	if ( size < offset )
	{
		fprintf(stderr, "error reading axis on file %s\n", filename.toStdString().c_str());
		return 2;
	}
	FILE *pFile = fopen(filename.toStdString().c_str(), "r");
	if ( pFile == NULL ) return 0;
	char str[offset];
	fseek(pFile, 0x4c, SEEK_SET);
	int n = 0, cont = 1;
	int typeAxe1 = -1;
	while ( cont )
	{
		int nbre = fscanf(pFile, "TYPE_AXE1=\t%d\n", &typeAxe1);
		if ( nbre > 0 )
			cont = 0;
		else
			fgets(str, offset, pFile);
		n++;
		if ( n > 20 )
		{
			cont = 0;
			strcpy(str, "Other");
		}
	}
	fclose(pFile);
	//if ( strcmp(str, "Time") == 0 ) return 0;
	//if ( strcmp(str, "Depth") == 0 ) return 1;
	if (typeAxe1==1) return 0;
	if (typeAxe1==2) return 1;
	return 2;
}

QString SeismisDatabaseManager::seismicGetNameFromPath(QString filename)
{
	QFileInfo fi(filename);
	QString ext = fi.suffix();
	QString descFilename = ProjectManagerNames::removeLastSuffix(filename) + ".desc";
	qDebug() << descFilename;
	if ( ext.compare("xt") == 0 )
	{
		QString name0 = ProjectManagerNames::getKeyFromFilename(descFilename, "name=");
		if ( !name0.isEmpty() )
		{
			return name0;
		}
	}
	QString tmp = fi.completeBaseName();
	QString header = tmp.left(10);
	if ( header.compare(QString("seismic3d.")) == 0 )
	{
		tmp.remove(0, 10);
	}
	return tmp;
}


void SeismisDatabaseManager::databaseWrite(QString filename, std::vector<SeismisDatabaseManager::Data> &vData)
{
	FILE *pFile = nullptr;
	fprintf(stderr, "database filename: %s\n", filename.toStdString().c_str());
	pFile = fopen(filename.toStdString().c_str(), "w");
	if ( pFile == nullptr ) return;

	int N = vData.size();
	fprintf(pFile, "Seismic database\n");
	fprintf(pFile, "Seismic number: %d\n", N);
	for (int n=0; n<vData.size(); n++)
	{
		QString color = "white";
		SeismisDatabaseManager::Data data = vData[n];
		if ( data.color == SeismicManager::TIME_SHORT_COLOR ) color = SeismicManager::TIME_SHORT_COLOR_STR;
			else if ( data.color == SeismicManager::TIME_8BIT_COLOR ) color = SeismicManager::TIME_8BIT_COLOR_STR;
			else if ( data.color == SeismicManager::TIME_32BIT_COLOR ) color = SeismicManager::TIME_32BIT_COLOR_STR;
			else if ( data.color == SeismicManager::DEPTH_SHORT_COLOR ) color = SeismicManager::DEPTH_SHORT_COLOR_STR;
			else if ( data.color == SeismicManager::DEPTH_8BIT_COLOR ) color = SeismicManager::DEPTH_8BIT_COLOR_STR;
			else if ( data.color == SeismicManager::DEPTH_32BIT_COLOR ) color = SeismicManager::DEPTH_32BIT_COLOR_STR;
			else if ( data.color == Qt::green ) color = QString("green");
		fprintf(pFile, "%d %s;%s;%s;%d;%d;%d\n", n, data.name.toStdString().c_str(), data.path.toStdString().c_str(), color.toStdString().c_str(), data.dimx, data.dimy, data.dimz);
	}
	fclose(pFile);
	chmod(filename.toStdString().c_str(), (mode_t)0777);
}

int SeismisDatabaseManager::dataBaseVersion(QString databasePath)
{
	FILE *pFile = NULL;
	pFile = fopen(databasePath.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return 0;

	char buff[1000];
	int N = 0;
	fscanf(pFile, "Seismic database\n", buff);
	fscanf(pFile, "Seismic number: %d\n", &N);
	char *res0 = fgets(buff, 100000, pFile);
	QString qres0(res0);
	std::vector<QString> str = seismicLineSplit(qres0);
	fclose(pFile);
	if ( str.size() == 4 ) return 1;
	else if ( str.size() == 7 ) return 2;
	return 0;
}



#include <stdio.h>

#include <QDebug>
#include <QProcess>
#include "Xt.h"
#include <iostream>
#include <sys/stat.h>
#include <seismicDatabaseManager.h>
#include <SeismicManager.h>
#include "globalconfig.h"


SeismicManager::SeismicManager(QWidget* parent) :
		ObjectManager(parent) {
	setContextMenu(true);
}

SeismicManager::~SeismicManager()
{

}



void SeismicManager::setSurveyPath(QString path, QString name)
{
	dataClear();
	dataBasketClear();
	m_surveyPath = path;
	m_surveyName = name;
	m_seismicPath = m_surveyPath + seismicRootSubDir;
	// qDebug() << "[ SEISMIC ]" << m_seismicPath;
	updateNames();
	displayNames();
	setSurveyName(name);
}

std::vector<QString> SeismicManager::getNames()
{
	return m_namesBasket.getTiny();
}

std::vector<QString> SeismicManager::getPath()
{
	std::vector<QString> ret;
	std::vector<QString> names = m_namesBasket.getFull();
	ret.resize(names.size());
	for (int i=0; i<names.size(); i++)
		ret[i] = names[i];
	return ret;
}

QString SeismicManager::getSeismicDirectory()
{
	return m_seismicPath;
}

std::vector<QString> SeismicManager::getAllNames()
{
	return m_names.getTiny();
}

std::vector<QString> SeismicManager::getAllPath()
{
	std::vector<QString> ret;
	std::vector<QString> names = m_names.getFull();
	ret.resize(names.size());
	for (int i=0; i<names.size(); i++)
		ret[i] = names[i];
	return ret;
}

std::vector<int> SeismicManager::getAllDimx()
{
	return m_names.dimx;
}

std::vector<int> SeismicManager::getAllDimy()
{
	return m_names.dimy;
}

std::vector<int> SeismicManager::getAllDimz()
{
	return m_names.dimz;
}

QString SeismicManager::getSurveyName()
{
	return m_surveyName;
}

QString SeismicManager::getSurveyPath()
{
	return m_surveyPath;
}


void SeismicManager::setForceBasket(std::vector<QString> tinyName0)
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




void SeismicManager::updateNames()
{
	QString db_filename = getDatabaseName();
	// qDebug() << "[DATABASE] :" << db_filename;

	if (  QFile::exists(db_filename) )
	{
		updateNamesFromDataBase(db_filename);
	}
	else
	{
		updateNamesFromDisk();
		saveSeismicListToDataBase(db_filename);
	}
}

void SeismicManager::f_dataBaseUpdate()
{
	/*
	QString db_filename = getDatabaseName();
	updateNamesFromDisk();
	saveSeismicListToDataBase(db_filename);
	displayNames();
	*/

	QString filename = getDatabaseName();
	QString datasetPath = getSeismicDirectory();
	SeismisDatabaseManager::update(filename, datasetPath);
	updateNames();
	displayNames();
}

void SeismicManager::updateNamesFromDisk()
{
    std::vector<QString> fullnames = getSeismicList(m_seismicPath);
    std::vector<QBrush> color;
    int N = fullnames.size();
    std::vector<QString> tiny, fullnamesOut;
    tiny.resize(N);
    color.resize(N);
    fullnamesOut.resize(N);
    int newListIndex = 0;

    for (int n=0; n<N; n++)
    {
    	bool isValid = true;
    	QBrush color0;
    	QString name0 = seismicFullFilenameToTinyName(fullnames[n]);
    	QFileInfo fi(fullnames[n]);
    	QString ext = fi.suffix();
    	if ( ext.compare("xt") == 0 )
    	{
    		int axis = filextGetAxis(fullnames[n]);
    		inri::Xt::Type type = inri::Xt::Unknown;
    		if (axis==0 || axis==1) {
    			inri::Xt xt(fullnames[n].toStdString().c_str());
    			if (xt.is_valid()) {
    				type = xt.type();
    			}
    		}
    		if (axis == 0 ) {
    			switch (type) {
    			case inri::Xt::Signed_16:
    				color0 = TIME_SHORT_COLOR;
    				break;
    			case inri::Xt::Unsigned_16:
    			case inri::Xt::Unsigned_8:
    			case inri::Xt::Signed_8:
    				color0 = TIME_8BIT_COLOR;
    				break;
    			default:
    				color0 = TIME_32BIT_COLOR;
    				break;
    			}
    		} else if (axis == 1 ) {
    			switch (type) {
				case inri::Xt::Signed_16:
					color0 = DEPTH_SHORT_COLOR;
					break;
				case inri::Xt::Unsigned_16:
				case inri::Xt::Unsigned_8:
				case inri::Xt::Signed_8:
					color0 = DEPTH_8BIT_COLOR;
					break;
				default:
					color0 = DEPTH_32BIT_COLOR;
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
    	tiny[newListIndex] = name0;
    	color[newListIndex] = color0;
    	fullnamesOut[newListIndex] = fullnames[n];
    	// qDebug() << name0 <<  " - " << fullnames[n];
        if (isValid) {
            newListIndex++;
        }
    }
    if (newListIndex<N) {
        tiny.resize(newListIndex);
        color.resize(newListIndex);
        fullnamesOut.resize(newListIndex);
    }
    m_names.copy(tiny, fullnamesOut, color);
}


void SeismicManager::updateNamesFromDataBase(QString db_filename)
{
	/*
	// fprintf(stderr, "--> %s\n", db_filename.toStdString().c_str());
	std::vector<QString> fullnames;
	std::vector<QBrush> color;
	std::vector<QString> tiny;

	int N = 0, n0 = 0;
	char buff[100000], buff2[10000], buff3[100000];
	FILE *pFile = NULL;
	pFile = fopen(db_filename.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return;

	fscanf(pFile, "Seismic database\n", buff);
	fscanf(pFile, "Seismic number: %d\n", &N);
	tiny.resize(N);
	fullnames.resize(N);
	color.resize(N);
	for (int n=0; n<N; n++)
	{
		fscanf(pFile, "%d %[^;];%[^;];%[^\n]\n", &n0, buff, buff2, buff3);
		tiny[n] = QString(buff);
		fullnames[n] = QString(buff2);
		QBrush brush = Qt::white;
		if ( strcmp(buff3, TIME_SHORT_COLOR_STR.toStdString().c_str()) == 0 ) brush = TIME_SHORT_COLOR;
		if ( strcmp(buff3, TIME_8BIT_COLOR_STR.toStdString().c_str()) == 0 ) brush = TIME_8BIT_COLOR;
		if ( strcmp(buff3, TIME_32BIT_COLOR_STR.toStdString().c_str()) == 0 ) brush = TIME_32BIT_COLOR;
		if ( strcmp(buff3, DEPTH_SHORT_COLOR_STR.toStdString().c_str()) == 0 ) brush = DEPTH_SHORT_COLOR;
		if ( strcmp(buff3, DEPTH_8BIT_COLOR_STR.toStdString().c_str()) == 0 ) brush = DEPTH_8BIT_COLOR;
		if ( strcmp(buff3, DEPTH_32BIT_COLOR_STR.toStdString().c_str()) == 0 ) brush = DEPTH_32BIT_COLOR;
		if ( strcmp(buff3, "green") == 0 ) brush = Qt::green;
		color[n] = brush;
	}
	fclose(pFile);
	m_names.copy(tiny, fullnames, color);
	*/
	std::vector<SeismisDatabaseManager::Data> data0 = SeismisDatabaseManager::databaseRead(db_filename);
	std::vector<QString> fullnames;
	std::vector<QBrush> color;
	std::vector<QString> tiny;
	std::vector<int> dimx;
	std::vector<int> dimy;
	std::vector<int> dimz;
	int N = data0.size();
	tiny.resize(N);
	fullnames.resize(N);
	color.resize(N);
	dimx.resize(N, 0);
	dimy.resize(N, 0);
	dimz.resize(N, 0);
	for (int i=0; i<N; i++)
	{
		tiny[i] = data0[i].name;
		fullnames[i] = data0[i].path;
		color[i] = data0[i].color;
		dimx[i] = data0[i].dimx;
		dimy[i] = data0[i].dimy;
		dimz[i] = data0[i].dimz;
	}
	m_names.copy(tiny, fullnames, color, dimx, dimy, dimz);
}



void SeismicManager::saveSeismicListToDataBase(QString db_filename)
{
	FILE *pFile = NULL;
	fprintf(stderr, "database filename: %s\n", db_filename.toStdString().c_str());
	pFile = fopen(db_filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	std::vector<QString> full = m_names.getFull();
	std::vector<QString> tiny = m_names.getTiny();
	std::vector<QBrush> color = m_names.getColor();

	int N = full.size();
	fprintf(pFile, "Seismic database\n");
	fprintf(pFile, "Seismic number: %d\n", N);
	for (int n=0; n<N; n++)
	{
		QString seismic_color = "white";
		if ( color.size() > 0 )
		{
			if ( color[n] == TIME_SHORT_COLOR ) seismic_color = TIME_SHORT_COLOR_STR;
			else if ( color[n] == TIME_8BIT_COLOR ) seismic_color = TIME_8BIT_COLOR_STR;
			else if ( color[n] == TIME_32BIT_COLOR ) seismic_color = TIME_32BIT_COLOR_STR;
			else if ( color[n] == DEPTH_SHORT_COLOR ) seismic_color = DEPTH_SHORT_COLOR_STR;
			else if ( color[n] == DEPTH_8BIT_COLOR ) seismic_color = DEPTH_8BIT_COLOR_STR;
			else if ( color[n] == DEPTH_32BIT_COLOR ) seismic_color = DEPTH_32BIT_COLOR_STR;
			else if ( color[n] == Qt::green ) seismic_color = QString("green");
		}

		fprintf(pFile, "%d %s;%s;%s\n", n, tiny[n].toStdString().c_str(), full[n].toStdString().c_str(), seismic_color.toStdString().c_str());
	}
	fclose(pFile);
	chmod(db_filename.toStdString().c_str(), (mode_t)0777);
}




// fullpath + name
std::vector<QString> SeismicManager::getSeismicList(QString path)
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

    int N = list.size();
    fullName.resize(N);
    tinyName.resize(N);
    for (int i=0; i<list.size(); i++)
    {
        QFileInfo fileInfo = list[i];
        QString filename = fileInfo.fileName();
        fullName[i] = fileInfo.absoluteFilePath();
    }
    return fullName;
}


QString SeismicManager::seismicFullFilenameToTinyName(QString filename)
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


QString SeismicManager::seismicTinyNameModify(QString fullname, QString tinyname)
{


	return "";
}


int SeismicManager::filextGetAxis(QString filename)
{
	QProcess process;
	QStringList options;
	options << filename;
	process.start("TestXtFile", options);
	process.waitForFinished();

	if (process.exitCode()!=QProcess::NormalExit) {
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






QString SeismicManager::getDatabaseName()
{
	// qDebug() << "[project name]" << getProjectName();
	// return getDatabasePath() + "database_seismic_" + getProjIndexNameForDataBase() + QString("_") + getProjectName() + QString("_") + getSurveyName() + ".txt";

	GlobalConfig& config = GlobalConfig::getConfig();
	QString tmp = formatDirPath(m_seismicPath);
	tmp.replace("/", "_@_");
	return config.databasePath() + QString("/databasev2_seismic_") + tmp + ".txt";
}




/*
void SeismicManager::displayNames()
{
	listwidgetList->clear();
	std::vector<QString> full = m_names.getFull();
	std::vector<QString> tiny = m_names.getTiny();
	std::vector<QBrush> color = m_names.getColor();
	QString prefix = lineedit_search->text();
	for (int n=0; n<tiny.size(); n++)
	{
		QString str = tiny[n];
		if ( ProjectManagerNames::isMultiKeyInside(str, prefix ) )
		{
			QListWidgetItem *item = new QListWidgetItem;
			item->setText(str);
			item->setToolTip(str);
			if ( color.size() > n )
				item->setForeground(color[n]);
			this->listwidgetList->addItem(item);
		}
	}
}
*/


const QString SeismicManager::TIME_SHORT_COLOR_STR = "cyan";
const QString SeismicManager::TIME_SHORT_COLOR_QCOLOR_STR = "#339af0";
const QString SeismicManager::TIME_8BIT_COLOR_STR = "#a5d8ff";
const QString SeismicManager::TIME_32BIT_COLOR_STR = "#1971c2";
const QString SeismicManager::DEPTH_SHORT_COLOR_STR = "red";
const QString SeismicManager::DEPTH_SHORT_COLOR_QCOLOR_STR = "#ff6b6b";
const QString SeismicManager::DEPTH_8BIT_COLOR_STR = "#ffc9c9";
const QString SeismicManager::DEPTH_32BIT_COLOR_STR = "#e03131";

const QColor SeismicManager::TIME_SHORT_COLOR = QColor(TIME_SHORT_COLOR_QCOLOR_STR);
const QColor SeismicManager::TIME_8BIT_COLOR = QColor(TIME_8BIT_COLOR_STR);
const QColor SeismicManager::TIME_32BIT_COLOR = QColor(TIME_32BIT_COLOR_STR);
const QColor SeismicManager::DEPTH_SHORT_COLOR = QColor(DEPTH_SHORT_COLOR_QCOLOR_STR);
const QColor SeismicManager::DEPTH_8BIT_COLOR = QColor(DEPTH_8BIT_COLOR_STR);
const QColor SeismicManager::DEPTH_32BIT_COLOR = QColor(DEPTH_32BIT_COLOR_STR);


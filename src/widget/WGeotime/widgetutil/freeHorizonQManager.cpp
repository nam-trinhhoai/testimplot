
#include <QDir>
#include <QFileInfoList>
#include <QPainter>
#include <QRegularExpression>
#include <QStringList>
#include <seismic3ddataset.h>
#include <freeHorizonManager.h>
#include <freeHorizonQManager.h>

int FreeHorizonQManager::getIsoFromDirectory(QString dir)
{
	// int pos = filename.lastIndexOf(QChar('_'));
	// int pos = dir.indexOf(QChar('_'));
	QString txt = dir.right(5);
	return txt.toInt();
}


QString FreeHorizonQManager::getPrefixFromFile(QString filename)
{
	// int pos = filename.lastIndexOf(QChar('_'));
	if ( filename == "isochrone.iso" ) return "isochrone";
	int pos = filename.indexOf(QChar('_'));
	return filename.left(pos);
}

// todo
QString FreeHorizonQManager::getRgtDataSetNameFromPath(QString path)
{
	QString path0 = path.replace("//", "/");
	int pos = path0.lastIndexOf("/ImportExport/IJK/HORIZON");
	path0 = path0.left(pos);
	// qDebug() << path0;
	return path0;
}

QString FreeHorizonQManager::getDataSetNameFromPath(QString path)
{
	/*
	QString path0 = path.replace("//", "/");
	int pos = path0.lastIndexOf("/ImportExport/IJK/HORIZON");
	path0 = path0.left(pos);
	pos = path0.replace("_(", "");
	pos = path0.replace(")", "");
	// qDebug() << path0;
	return path0;
	*/
	return "";
}



std::vector<QString> FreeHorizonQManager::getListName(QString path)
{
	std::vector<QString> out;

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

std::vector<QString> FreeHorizonQManager::getListPath(QString path)
{
	std::vector<QString> out;

	QDir mainDir(path);
	if ( !mainDir.exists() ) return out;
	QFileInfoList mainList = mainDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable | QDir::Executable);
	if ( mainList.size() == 0 ) return out;
	out.resize(mainList.size());

	for (int i=0; i<mainList.size(); i++)
	{
		out[i] = mainList[i].absoluteFilePath() + "/";
	}
	return out;
}


std::vector<QString> FreeHorizonQManager::getDataSet(QString horizonPath)
{
	std::vector<QString> out;
	QDir dir(horizonPath);
	if ( !dir.exists() ) return out;

	for (int i=0; i<FreeHorizonManager::tabAttributSuffix.size(); i++)
	{
		QString crit = "*" + QString::fromStdString(FreeHorizonManager::tabAttributSuffix[i]) + QString::fromStdString(FreeHorizonManager::attributExt);
		QFileInfoList list0 = dir.entryInfoList(QStringList() << crit, QDir::Files);
		for (int n=0; n<list0.size(); n++)
		{
			QString attributName = list0[n].baseName();
			QString dataSetName = getPrefixFromFile(attributName);
			out.push_back(dataSetName);
			std::sort( out.begin(), out.end() );
			out.erase( std::unique( out.begin(), out.end() ), out.end() );
		}
	}
	return out;
}



QColor FreeHorizonQManager::loadColorFromPath(const QString& path, bool* ok) {
	QString filename = path + "/color.txt";
	FILE* file = fopen(filename.toStdString().c_str(), "r");
	bool valid = file != nullptr;
	QColor loadedColor = Qt::white;
	if (valid) {
		char buff[4096];
		fscanf(file, "color file version 1.0\n", buff);

		int matchNumber = fscanf(file, "color: %[^\n]\n", buff);
		valid = matchNumber==1;
		if (valid) {
			loadedColor = QColor(buff);
			valid = loadedColor.isValid();
		}
		fclose(file);
	}
	if (ok!=nullptr) {
		*ok = valid;
	}
	return loadedColor;
}

bool FreeHorizonQManager::saveColorToPath(const QString& path, const QColor& color) {
	QString filename = path + "/color.txt";
	FILE* file = fopen(filename.toStdString().c_str(), "w");
	bool valid = file != nullptr;
	if (valid) {
		QString colorName = color.name();
		QString textToWrite = "color file version 1.0\ncolor: " + colorName + "\n";
		QByteArray colorBuf = textToWrite.toUtf8();
		fwrite(colorBuf.data(), sizeof(char), colorBuf.size(), file);
		fclose(file);
	}
	return valid;
}


std::vector<QString> FreeHorizonQManager::getAttributData(QString& path)
{
	std::vector<QString> list;
	QDir dir(path);
	QFileInfoList list0 = dir.entryInfoList(QDir::Files);
	for (int i=0; i<list0.size(); i++)
	{
		QString ext = list0[i].completeSuffix();
		if ( ext == "amp" || ext == "iso" || ext == "raw" )
		{
			QString filename = list0[i].fileName();
			list.push_back(filename);
		}
	}
	return list;
}

QIcon FreeHorizonQManager::getHorizonIcon(QColor color, SampleUnit sampleUnit, int size)
{
	QColor sampleColor = color;
	if (sampleUnit==SampleUnit::TIME)
	{
		sampleColor = Qt::blue;
	}
	else if (sampleUnit==SampleUnit::DEPTH)
	{
		sampleColor = Qt::red;
	}

	QImage img(size, size, QImage::Format_RGB32);
	QPainter p(&img);
	p.fillRect(img.rect(), Qt::white);

	p.setBrush(sampleColor);
	p.drawPolygon(QPolygon() << QPoint(0,0) << QPoint(0,size-1) << QPoint(size-1,0));
	p.setBrush(color);
	p.drawPolygon(QPolygon() << QPoint(size-1,size-1) << QPoint(1,size-1) << QPoint(size-1,1));

	QPixmap pixmap = QPixmap::fromImage(img);
	return QIcon(pixmap);
}

QIcon FreeHorizonQManager::getHorizonIcon(QString path, int size)
{
	bool ok = true;
	QColor color = FreeHorizonQManager::loadColorFromPath(path, &ok);
	inri::Xt::Axis axis = FreeHorizonManager::dataSetGetAxis(path.toStdString());
	SampleUnit sampleUnit;
	if ( axis == inri::Xt::Axis::Depth ) sampleUnit = SampleUnit::DEPTH;
	else if ( axis == inri::Xt::Axis::Time ) sampleUnit = SampleUnit::TIME;
	else sampleUnit = SampleUnit::NONE;
	return getHorizonIcon(color, sampleUnit, size);
}

std::vector<QString> FreeHorizonQManager::getAttributPath(QString& path)
{
	std::vector<QString> list;
	QDir dir(path);
	QFileInfoList list0 = dir.entryInfoList(QDir::Files);
	for (int i=0; i<list0.size(); i++)
	{
		QString ext = list0[i].completeSuffix();
		if ( ext == "amp" || ext == "iso" )
		{
			QString filename = list0[i].filePath();
			list.push_back(filename);
		}
	}
	return list;
}


// todo
// rename class ?

QIcon FreeHorizonQManager::getDataSetIcon(QString path) {
	QIcon icon0;
	QFile file(path);
	if ( !file.exists() ) return icon0;
	inri::Xt xt((char*)path.toStdString().c_str());
	if ( !xt.is_valid() ) return icon0;

	QString typeName = "";
	inri::Xt::Type type = xt.type();
	bool typeFound = true;
	if ( type == inri::Xt::Type::Signed_8 )
	{
		typeName = "Clair";
	}
	else if ( type == inri::Xt::Type::Signed_16 )
	{
		typeName = "Vif";
	}
	else if ( type == inri::Xt::Type::Unsigned_32 || type == inri::Xt::Type::Float )
	{
		typeName = "Foncé";
	}
	else
	{
		typeFound = false;
	}

	QString unitName;
	bool unitFound = true;
	inri::Xt::Axis axis = xt.axis();
	if ( axis == inri::Xt::Axis::Depth )
	{
		unitName = "Rouge";
	}
	else if ( axis == inri::Xt::Axis::Time )
	{
		unitName = "Bleu";
	}
	else
	{
		unitFound = false;
	}

	Seismic3DDataset::CUBE_TYPE cubeType = (path.contains(QRegularExpression("_[rR][gG][tT]"))) ? Seismic3DDataset::CUBE_TYPE::RGT:Seismic3DDataset::CUBE_TYPE::Seismic;
	if (cubeType==Seismic3DDataset::CUBE_TYPE::Seismic) {
		cubeType = (path.contains(QRegularExpression("[nN][eE][xX][tT][vV][iI][sS][iI][oO][nN][pP][aA][tT][cC][hH]"))) ? Seismic3DDataset::CUBE_TYPE::Patch:Seismic3DDataset::CUBE_TYPE::Seismic;
	}
	QString cubeTypeName;
	if (cubeType==Seismic3DDataset::CUBE_TYPE::RGT) {
		cubeTypeName = ".Vert";
	} else if (cubeType==Seismic3DDataset::CUBE_TYPE::Patch) {
		cubeTypeName = ".Jaune";
	}
	bool iconFound = typeFound && unitFound;
	if (iconFound) {
		QString iconFile = ":/slicer/icons/dataset_icons/" + unitName + typeName + cubeTypeName + ".svg";
		QIcon icon(iconFile);
		return icon;
	}
	return icon0;
}


QString FreeHorizonQManager::getSizeOnDisk(QString path)
{
	if ( path == "" ) return "";
	FILE* f = fopen((char*)path.toStdString().c_str(), "r");
	std::size_t size = 0;
	if (f!=nullptr) {
		fseek(f, 0L, SEEK_END);
		size = ftell(f);
		fclose(f);
	}
	else
	{
		return "unknown";
	}
	if ( size < 1000 ) return QString::number(size) + " bytes";
	if ( size < 1000000 ) return QString::number((double)size/1000.0) + " kbytes";
	if ( size < 1000000000) return QString::number((double)size/1000000.0) + " Mbytes";
	return QString::number((double)size/1000000000.0) + " Gbytes";
}


// ====================== iso

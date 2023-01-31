
#include <QProcess>
#include <QString>
#include <QFile>
#include <QDebug>
#include <QIODevice>
#include <QTime>
#include <QTextStream>
#include <QDate>
#include <QDir>
#include <QFileInfo>

#include <dataFileCreationInfo.h>

DataFileCreationInfo::DataFileCreationInfo()
{

}


DataFileCreationInfo::~DataFileCreationInfo()
{

}

DataFileCreationInfo::DataFileCreationInfo(QString path)
{
	setDataPath(path);
}


void DataFileCreationInfo::setDataPath(QString path)
{
	m_dataPath = path;
}

void DataFileCreationInfo::setDataName(QString name)
{
	m_dataName = name;
}

// =========================================================
QString DataFileCreationInfo::getDirectoryPath()
{
	// QFileInfo fi(m_dataPath);
	QDir dir(m_dataPath);
	return dir.filePath(m_dataPath);
	// return d.filePath(m_dataPath);
}

QString DataFileCreationInfo::getFilename()
{
	QFileInfo fi(m_dataPath);
	return fi.fileName();
}

QString DataFileCreationInfo::getCompleteBaseName()
{
	QFileInfo fi(m_dataPath);
	return fi.completeBaseName();
}


QString DataFileCreationInfo::getUserName()
{
	QProcess process;
	process.start("whoami");
	process.waitForFinished();
	QString name = "";
	if (process.exitCode()==QProcess::NormalExit) {
		name = process.readAllStandardOutput();
	} else {
		name = "unknown";
	}
	return name;
}

QString DataFileCreationInfo::currentTime()
{
	QTime time = QTime::currentTime();
	QString formattedTime = time.toString("hh:mm:ss");
	return formattedTime;
}

QString DataFileCreationInfo::currentDate()
{
	QDate cd = QDate::currentDate();
	return cd.toString(Qt::TextDate);
}

void DataFileCreationInfo::clear()
{

}

QString DataFileCreationInfo::header()
{
	QString header = "data info v1.0";
	return header;
}

void DataFileCreationInfo::addComment(QString type, int val)
{
	addComment(type, QString::number(val));
}

void DataFileCreationInfo::addComment(QString type, QString val)
{
	QString msg = type + ": " + val;
	m_message.push_back(msg);
}

void DataFileCreationInfo::addComment(QString type, double val)
{
	addComment(type, QString::number(val));
}

void DataFileCreationInfo::addComment(QString type, long val)
{
	addComment(type, QString::number(val));
}

void DataFileCreationInfo::addComment(QString type, std::string val)
{
	addComment(type, QString::fromStdString(val));
}


QString DataFileCreationInfo::getMessagePath()
{
	QString path = "";
	QString path0 = getDirectoryPath();
	QString name = getCompleteBaseName() + ".txt";
	QDir dir(path0);
	dir.cdUp();
	dir.cdUp();
	path = path0 + "/ImportExport/IJK/FileInfo/";
	QDir d;
	d.mkpath(path);
	return path + name;
}

std::string DataFileCreationInfo::write()
{
	QString msgPath = getMessagePath();

	qDebug() << msgPath;
	return "";


//	QFile file(msgPath);
//	if ( !file.open(QIODevice::WriteOnly | QIODevice::Text) )
//	{
//		qDebug() << "Unable to create info file: " << msgPath;
//		return "fail";
//	}
//
//	QTextStream out(&file);
//	out << header() << "\n";
//	out << "user name: " << getUserName() << "\n";
//	out << "date: " << currentDate() << " [ " << currentDate() << " ]" << "\n";
//	for (QString str : m_message)
//		out << str << "\n";
//
//	return "ok";
}



#include <sys/stat.h>
#include <QDir>
#include <QTemporaryFile>


#include <QFileUtils.h>

bool mkdirPathIfNotExist(QString path)
{
	QDir dir0(path);
	if ( !dir0.exists() )
	{
		QDir dir;
		// dir.mkdir(path, 0x7776);
		// QFile::Permissions perm = QFileDevice::ReadOwner|QFileDevice::WriteOwner|QFileDevice::ReadUser|QFileDevice::WriteUser|QFileDevice::ReadGroup|QFileDevice::WriteGroup|QFileDevice::ReadOther|QFileDevice::WriteOther;
		dir.mkdir(path);
	}
	QDir dir1(path);
	if ( !dir1.exists() ) return false;
	chmod((char*)path.toStdString().c_str(), (mode_t)0777);
	return true;
}


QString makeFilePathWithSuffix(QString source, QString suffix, QString ext)
{
	int lastPoint = source.lastIndexOf(".");
	QString prefix = source.left(lastPoint);
	QString out = prefix + "_" + suffix + ext;
	return out;
}


QString getDirectoryUp(QString dir0)
{
	int lastPoint = dir0.lastIndexOf("/");
	QString dir = dir0.left(lastPoint);
	return dir;
}

QString getTmpFilename(QString dirPath, QString prefix, QString ext)
{
	QDir tempDir = QDir(dirPath);
	tempDir.mkpath(".");
	QTemporaryFile outFile;
	QString format = prefix + "_XXXXXX" + ext;
	outFile.setFileTemplate(tempDir.absoluteFilePath(format));
	outFile.setAutoRemove(false);
	outFile.open();
	outFile.close();
	return outFile.fileName();
}


bool fileRemove(QString filename)
{
	QFile file (filename);
	return file.remove();
}

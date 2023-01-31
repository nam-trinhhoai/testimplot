
#include <isoHorizonManager.h>
#include <isoHorizonQManager.h>

#include <QDir>
#include <QFileInfoList>

std::vector<QString> IsoHorizonQManager::getListDir(QString path)
{
	QDir dir(path);
    dir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
    dir.setSorting(QDir::Name);
    QFileInfoList list = dir.entryInfoList();

    std::vector<QString> ret;
    ret.resize(list.size(), "");
    for (int i=0; i<list.size(); i++)
    	ret[i] = list[i].absoluteFilePath();
    return ret;
}



bool IsoHorizonQManager::isAttributComputed(QString mainPath, std::vector<std::string> isoDir, QString attributName)
{
	for (int i=0; i<isoDir.size(); i++)
	{
		QString filename = mainPath + "/" + QString::fromStdString(isoDir[i]) + "/" + attributName;
		QFile fi(filename);
		if ( !fi.exists() ) return false;
	}
	return true;
}

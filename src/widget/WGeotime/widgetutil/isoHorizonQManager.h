

#ifndef __ISOHORIZONQMANAGER__
#define __ISOHORIZONQMANAGER__

#include <vector>
#include <QString>


class IsoHorizonQManager
{
public:
	int no = 0;
	static std::vector<QString> getListDir(QString path);
	static bool isAttributComputed(QString mainPath, std::vector<std::string> isoDir, QString attributName);

};




#endif

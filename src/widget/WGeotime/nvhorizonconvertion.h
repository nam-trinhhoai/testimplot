
#ifndef __NVHORIZONCONVERSION__
#define __NVHORIZONCONVERSION__

#include <QString>

class GeotimeProjectManagerWidget;
class ProjectManagerWidget;

class NVHorizonConvertion
{
public:
	static bool convertion(GeotimeProjectManagerWidget* manager);
	static bool convertion(ProjectManagerWidget* manager);

private:
	static bool convertion(QString oldPath, QString newPath);
	static bool directoryConvertion(QString oldBasePath, QString oldName, QString newBasePath);
	static QString newNameGet(QString oldName);
	static bool directoryFileConvertion(QString oldPath, QString newPath);
	static std::vector<QString> getFiles(QString path);
	static bool attributConversion(QString iso, QString src, QString dst);
	static int getDimvfromRaw(QString filename, int dimy, int dimz);

};


#endif

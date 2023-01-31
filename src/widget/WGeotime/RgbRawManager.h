#ifndef __RGBRAWMANAGER__
#define __RGBRAWMANAGER__



#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QListWidget>
#include <QDir>
#include <QLineEdit>
#include <QTabWidget>
#include <QGroupBox>
#include <QTableWidget>
#include <QPushButton>
#include <QVBoxLayout>

#include <utility>

#include <ObjectManager.h>
#include <ProjectManagerNames.h>



class RgbRawManager : public ObjectManager{

public:
	RgbRawManager(QWidget* parent = 0);
	virtual ~RgbRawManager();
	void setSurvey(QString path, QString name);
	void setForceBasket(std::vector<QString> tinyName, std::vector<QString> fullName);
	QString m_surveyPath, m_surveyName, m_RgbRawPath;
	std::vector<QString> directoryName;
	std::vector<QString> directoryPath;
	std::vector<QString> getNames();
	std::vector<QString> getPath();
	std::vector<QString> getAllDirectoryNames();
	std::vector<QString> getAllDirectoryPath();
	void updateNames();
	std::vector<QString> getAviNames();
	std::vector<QString> getAviPath();


	// QString getSurveyName();
	// QString getSurveyPath();


private:
	// ProjectManagerNames m_names;
	const QString RgbRawRootSubDir = "ImportExport/IJK/";
	const QString RgbRawRootSubDir2 = "cubeRgt2RGB/";
	ProjectManagerNames m_aviNames;
	// void displayNames();
	// std::vector<QString> getSeismicList(QString path);
	// QString seismicFullFilenameToTinyName(QString filename);
	// QString seismicTinyNameModify(QString fullname, QString tinyname);
	QString getDatabaseName();
	void updateNamesFromDataBase(QString db_filename);
	void updateNamesFromDisk();
	void updateDirectoryNames();
	void saveListToDataBase(QString db_filename);

};





#endif

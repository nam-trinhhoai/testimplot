
#ifndef __SEISMICMANAGER__
#define __SEISMICMANAGER__

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



class SeismicManager : public ObjectManager{

public:
	SeismicManager(QWidget* parent = 0);
	virtual ~SeismicManager();
	void setSurveyPath(QString path, QString name);
	void setForceBasket(std::vector<QString> tinyName);
	QString m_surveyPath, m_seismicPath;
	std::vector<QString> getNames();
	std::vector<QString> getPath();
	std::vector<QString> getAllNames();
	std::vector<QString> getAllPath();
	std::vector<int> getAllDimx();
	std::vector<int> getAllDimy();
	std::vector<int> getAllDimz();

	QString getSurveyName();
	QString getSurveyPath();
	QString getSeismicDirectory();

	static const QString TIME_SHORT_COLOR_STR;
	static const QString TIME_SHORT_COLOR_QCOLOR_STR;
	static const QString TIME_8BIT_COLOR_STR;
	static const QString TIME_32BIT_COLOR_STR;
	static const QString DEPTH_SHORT_COLOR_STR;
	static const QString DEPTH_SHORT_COLOR_QCOLOR_STR;
	static const QString DEPTH_8BIT_COLOR_STR;
	static const QString DEPTH_32BIT_COLOR_STR;

	static const QColor TIME_SHORT_COLOR;
	static const QColor TIME_8BIT_COLOR;
	static const QColor TIME_32BIT_COLOR;
	static const QColor DEPTH_SHORT_COLOR;
	static const QColor DEPTH_8BIT_COLOR;
	static const QColor DEPTH_32BIT_COLOR;

	static std::vector<QString> getSeismicList(QString path); // get seismics lists
	static QString seismicFullFilenameToTinyName(QString filename); // get tiny name
	static int filextGetAxis(QString filename); // get axis and also check for validity in the function

	void f_dataBaseUpdate();

private:
	// ProjectManagerNames m_names;
	const QString seismicRootSubDir = "DATA/SEISMIC/";
	void updateNames();
	// void displayNames();
	QString seismicTinyNameModify(QString fullname, QString tinyname);
	QString getDatabaseName();
	void updateNamesFromDataBase(QString db_filename);
	void updateNamesFromDisk();
	void saveSeismicListToDataBase(QString db_filename);

};






#endif

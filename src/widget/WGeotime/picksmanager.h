
#ifndef __PICKSMANAGER__
#define __PICKSMANAGER__

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
#include <QColor>

#include <utility>

#include <ObjectManager.h>
#include <ProjectManagerNames.h>



class PicksManager : public ObjectManager{

public:
	PicksManager(QWidget* parent = 0);
	virtual ~PicksManager();
	void setProjectPath(QString path, QString name);
	std::vector<QString> getNames();
	std::vector<QString> getPath();
	std::vector<QBrush> getColors();
	std::vector<QString> getAllNames();
	std::vector<QString> getAllPath();
	std::vector<QBrush> getAllColors();


private:
	const QString picksRootSubDir = "DATA/PICK_FEATURES/";
	QString m_projectName, m_projectPath, m_picksPath;
//	ProjectManagerNames m_picksName, m_picksBasket;
	QString getDatabaseName();
	void updateNames();
	void updateNamesFromDisk();
	void updateNamesFromDataBase(QString db_filename);
	void saveListToDataBase(QString db_filename);
	QString fullFilenameToTinyName(QString filename);
	void f_dataBaseUpdate();
	QColor fullFilenameToColor(QString filename);

	std::vector<QBrush> m_allColors;


	/*
	void updateNames();
	void welllogNamesUpdate(QString path);
	// void displayNamesBasket();
	void displayNamesBasket0();
	void f_basketAdd();
	void f_basketSub();
	void f_basketListClick(QListWidgetItem* listItem);
	void f_basketListClick(QString wellTinyName);
	*/






	/*
	// void displayNames();
	std::vector<QString> getSeismicList(QString path);
	QString seismicFullFilenameToTinyName(QString filename);
	QString seismicTinyNameModify(QString fullname, QString tinyname);
	int filextGetAxis(QString filename);
	*/
};





#endif

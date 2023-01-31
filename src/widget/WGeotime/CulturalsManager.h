
#ifndef __CULTURALSMANAGER__
#define __CULTURALSMANAGER__


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

#include <ObjectManager.h>
#include <ProjectManagerNames.h>



class CulturalsManager : public ObjectManager{

public:
	CulturalsManager(QWidget* parent = 0);
	virtual ~CulturalsManager();
	void setProjectPath(QString path, QString name);
	void setForceBasket(std::vector<QString> cdatName, std::vector<QString> cdatPath, std::vector<QString> strdName, std::vector<QString> strdPath);
	std::vector<QString> getCdatNames();
	std::vector<QString> getCdatPath();
	std::vector<QString> getStrdNames();
	std::vector<QString> getStrdPath();
	void f_basketAdd();


private:
	const QString culturalsRootSubDir = "DATA/CULTURAL/";
	QString m_projectName, m_projectPath, m_culturalsPath;
	ProjectManagerNames m_cdat, m_strd, m_cdatBasket, m_strdBasket;
	void updateNames();
	ProjectManagerNames updateNames(QString ext);
	// void displayNames();
	std::vector<QString> getCulturalsList(QString path);
	QString getDatabaseName();
	void updateNamesFromDisk();
	void updateNamesFromDataBase(QString db_filename);
	void saveListToDataBase(QString db_filename);
	void f_basketSub();
};




#endif

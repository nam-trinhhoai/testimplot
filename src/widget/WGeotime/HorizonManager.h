
#ifndef __HORIZONMANAGER__
#define __HORIZONMANAGER__

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



class HorizonManager : public ObjectManager{

public:
	HorizonManager(QWidget* parent = 0);
	virtual ~HorizonManager();
	void setSurveyPath(QString path, QString name);
	std::vector<QString> getNames();
	std::vector<QString> getPath();
	std::vector<QString> getAllNames();
	std::vector<QString> getAllPath();
	void setForceBasket(std::vector<QString> tinyName0);
	std::vector<QString> getIsoValueListName();
	std::vector<QString> getIsoValueListPath();
	std::vector<QString> getFreeName();
	std::vector<QString> getFreePath();

private:
	const QString horizonRootSubDir = "/ImportExport/IJK/";
	const QString horizonRootSubDirV2 = "/ImportExport/IJK/HORIZONS/";
	QString m_surveyName, m_surveyPath, m_horizonPath;
	void updateNames();
	void updateNamesFromDisk();
	void updateNamesFromDataBase(QString db_filename);
	void saveListToDataBase(QString db_filename);
	QString getDatabaseName();
	void f_dataBaseUpdate() override;
};




#endif




#ifndef __WELLSMANAGER__
#define __WELLSMANAGER__


#include <vector>

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

#include <vector>
#include <math.h>
#include <utility>

#include <WellsLogManager.h>
//#include <WellsPicksManager.h>
#include<WellsTF2PManager.h>
#include <WellsHeadManager.h>

#include <WellUtil.h>




class WellsManager : public QWidget{
	Q_OBJECT
public:
	WellsManager(QWidget* parent = 0);
	virtual ~WellsManager();
	void setProjectPath(QString path, QString name);
	void setProjectType(int type);
	void setForceDataBasket(WELLMASTER _data0);
	WellsHeadManager *m_wellsHeadManager;
	WellsLogManager *m_wellsLogManager;
	//WellsPicksManager *m_wellsPicksManager;
	WellsTF2PManager *m_wellsTF2PManager;
	WELLMASTER getBasket();
	int getWellsHeadBasketSelectedIndex();
	QString getWellsHeadBasketSelectedName();
	std::vector<WELLHEADDATA> getMainData();
	// std::vector<WELLLIST2> getPicksSortedWells(std::vector<QString> picksName);
	std::vector<MARKER> getPicksSortedWells(const std::vector<QString>& picksName, const std::vector<QBrush>& colors);

	std::vector<std::vector<QString>> getWellBasketLogPicksTf2pNames(QString type, QString nameType);
	std::vector<QString> getWellBasketTinyNames();
	std::vector<QString> getWellBasketFullNames();

public slots:
	void trt_dataBaseUpdate0();

private:
	std::vector<WELLHEADDATA> data0;
	const QString wellsRootSubDir = "DATA/WELLS/";
	const QString deviationFilename = "deviation";
	QTabWidget *tabwidget;
	// WellsHeadManager *m_wellsHeadManager;
	QString m_projectname, m_projectPath, m_wellPath;
	int m_projectType;
	void updateNames();
	void updateNamesFromDisk();
	void updateNamesFromDataBase(QString db_filename);
	void saveListToDataBase(QString db_filename);
	void wellsHeadDisplayNames();
	QString getDatabaseName();
	QString getDeviationNames(QString path);
	// std::pair<std::vector<QString>, std::vector<QString>> getLogNames(QString path);
	std::pair<std::vector<QString>, std::vector<QString>> getLogTF2PPicksNames(QString path, QString ext);
	void f_dataBaseUpdate();

private slots:
	void trt_debug();




};




#endif


#ifndef __WELLSHEADMANAGER__
#define __WELLSHEADMANAGER__

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
#include <WellUtil.h>
#include <WellsLogManager.h>
#include <WellsTF2PManager.h>
#include <WellsPicksManager.h>



class WellsHeadManager : public ObjectManager{

public:
	WellsHeadManager(QWidget* parent = 0);
	virtual ~WellsHeadManager();
	void setWellPath(QString path);
	std::vector<WELLHEADDATA> m_listData0;
	// ProjectManagerNames m_basketWell;
	// ProjectManagerNames m_basketWellBore;
	void setListData(std::vector<WELLHEADDATA> list);

	WellsLogManager *m_wellsLogManager = nullptr;
	WellsTF2PManager *m_wellsTF2PManager = nullptr;
	WellsPicksManager *m_wellsPicksManager = nullptr;
	void setWellsLogManager(WellsLogManager *wellsLogManager);
	void setWellsTF2PManager(WellsTF2PManager *wellsTF2PManager);
	void setWellsPicksManager(WellsPicksManager *wellsPicksManager);
	// std::vector<WELLLIST2> finalWellBasket;
	void setForceDataBasket(WELLMASTER _data0);
	WELLMASTER m_wellMaster;
	void displayNames();
	int getBasketSelectedIndex();
	QString getBasketSelectedName();

	std::vector<std::vector<QString>> getWellBasketLogPicksTf2pNames(QString type, QString nameType);
	std::vector<QString> getWellBasketTinyNames();
	std::vector<QString> getWellBasketFullNames();




private:
	const QString wellsHeadRootSubDir = "DATA/SEISMIC/";
	QString m_projectName, m_projectPath, m_wellsPath;
	void updateNames();
	void welllogNamesUpdate(QString path);
	// void displayNamesBasket();
	void displayNamesBasket0();
	void f_basketAdd();
	void f_basketSub();
//	void f_basketListClick(QListWidgetItem* listItem);
	void f_basketListClick(QString wellTinyName);
	void f_basketListSelectionChanged();
	void selectDefaultTFP(const std::vector<QString>& wellHead_tinyname_basket,
			const std::vector<QString>& wellHead_fullname_basket,
			const std::vector<QString>& wellBore_tinyname_basket,
			const std::vector<QString>& wellBore_fullname_basket);



	/*
	// void displayNames();
	std::vector<QString> getSeismicList(QString path);
	QString seismicFullFilenameToTinyName(QString filename);
	QString seismicTinyNameModify(QString fullname, QString tinyname);
	int filextGetAxis(QString filename);
	*/
};




#endif

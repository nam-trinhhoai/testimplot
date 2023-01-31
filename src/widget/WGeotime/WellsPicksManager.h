
#ifndef __WELLSPICKSMANAGER__
#define __WELLSPICKSMANAGER__

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
#include <WellUtil.h>
#include <ObjectManager.h>
#include <ProjectManagerNames.h>

class WellsManager;

class WellsPicksManager : public ObjectManager{

public:
	WellsPicksManager(QWidget* parent = 0);
	virtual ~WellsPicksManager();
	void setFinalWellbasket(std::vector<WELLLIST> *data);
	void setWellHeadBasket(ProjectManagerNames *names);
	void setWellBoreHeadBasket(ProjectManagerNames *names);
	void setWellHeadNames(QString tiny, QString full);
	void setWellBoreNames(QString tiny, QString full);
	void displayNamesBasket();
	void f_basketAdd();
	void setWellManager(WellsManager *wellManager);

private:
	WellsManager *m_wellManager = nullptr;
	std::vector<WELLLIST> *m_finalWellBasket;
	ProjectManagerNames *m_wellHeadBasket;
	ProjectManagerNames *m_wellBoreHeadBasket;
	QString wellHeadTinyName;
	QString wellHeadFullName;
	QString wellBoreTinyName;
	QString wellBoreFullName;
	void f_basketSub();
	void getIndexFromPicksName(QString picks_displayname, int *idx_well, int *idx_bore, int *idx);

};




#endif

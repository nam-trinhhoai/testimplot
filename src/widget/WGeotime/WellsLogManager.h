
#ifndef __WELLSLOGMANAGER__
#define __WELLSLOGMANAGER__

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



class WellsLogManager : public ObjectManager{

public:
	WellsLogManager(QWidget* parent = 0);
	virtual ~WellsLogManager();
	void setFinalWellbasket(std::vector<WELLLIST> *data);
	void setWellHeadBasket(ProjectManagerNames *names);
	void setWellBoreHeadBasket(ProjectManagerNames *names);
	void setWellHeadNames(QString tiny, QString full);
	void setWellBoreNames(QString tiny, QString full);
	void displayNamesBasket();
	void f_basketAdd();
	void f_basketSub();

	void setWellManager(WellsManager *wellManager);


private:
	std::vector<WELLLIST> *m_finalWellBasket;
	ProjectManagerNames *m_wellHeadBasket;
	ProjectManagerNames *m_wellBoreHeadBasket;
	QString wellHeadTinyName;
	QString wellHeadFullName;
	QString wellBoreTinyName;
	QString wellBoreFullName;
	void getIndexFromLogName(QString log_displayname, int *idx_well, int *idx_bore, int *idx);
	WellsManager *m_wellManager = nullptr;

	void debug();
};




#endif


#ifndef __OBJECTMANAGER__
#define __OBJECTMANAGER__

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
#include <QPoint>
#include <QMenu>

#include <vector>
#include <math.h>

#include <ProjectManagerNames.h>

class ObjectManager : public QWidget{
	Q_OBJECT
public:
	ObjectManager(QWidget* parent = 0);
	virtual ~ObjectManager();
	QLineEdit *lineedit_search;
	QListWidget *listwidgetList, *listwidgetBasket;
	ProjectManagerNames m_names, m_namesBasket;

	virtual void displayClear();
	virtual void displayBasketClear();
	virtual void dataClear();
	virtual void dataBasketClear();
	void setLabelSearchVisible(bool val);
	void setLabelSearchText(QString txt);

	void setListMultiSelection(bool type);
	void setListBasketMultiSelection(QAbstractItemView::SelectionMode type);

	void setProjectType(int type);
	void setProjectName(QString name);
	void setSurveyName(QString name);
	void setProjectCustomPath(QString path);
	void setDatabasePath(QString new_path);
	void setContextMenu(bool val);

	int getProjectType();
	QString  getProjectName();
	QString getSurveyName();
	QString getProjectCustomPath();
	QString getDatabasePath();
	QList<QListWidgetItem*> getBasketSelectedItems();
	QString getBasketSelectedName();

	void setNames(ProjectManagerNames names);
	virtual void displayNames();
	virtual void setBasketNames(ProjectManagerNames names);
	virtual void displayNamesBasket();
	void basketAdd();
	void basketSub();
	void clearBasket();

	virtual void f_basketAdd();
	virtual void f_basketSub();
	virtual void f_basketListClick(QListWidgetItem* listItem);
	virtual void f_basketListSelectionChanged();
	virtual void f_dataBaseUpdate();

	QString getProjIndexNameForDataBase();
	QString m_surveyName;
	void setButtonDataBase(bool val);
	void setVisibleBasket(bool val);


private:
	QPushButton *pushbutton_add = nullptr;
	QPushButton *pushbutton_sub = nullptr;
	int m_projectType;
	QString m_projectName;
	QString m_projectCustomPath;
	QString m_dataBasePath;
	QPushButton *pushbutton_databaseUpdate;
	QLabel *labelSearchHelp;
	QString getFullNamefromTinyName(QString tinyName);
	void ProvideContextMenu(QListWidget *listWidget, const QPoint &pos);
	bool visibleBasket = true;
	void visibleBasketDisplay(bool val);


public slots:
	void trt_SearchChange(QString str);
	void trt_basketAdd();
	void trt_basketSub();
	void trt_basketListClick(QListWidgetItem* listItem);
	void trt_basketListSelectionChanged();
	void trt_dataBaseUpdate();

	void ProvideContextMenuList(const QPoint &pos);
	void ProvideContextMenubasket(const QPoint &pos);
	static QString formatDirPath(const QString& path_to_format);

// private:
	// QLineEdit *lineedit_search;
	// QListWidget *listwidgetList, *listwidgetBasket;
	// ProjectManagerNames *m_names;

};



#endif

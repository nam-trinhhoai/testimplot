
#include <QDebug>

#include<ProjectManagerNames.h>
#include <ObjectManager.h>
#include "globalconfig.h"
#include <fileInformationWidget.h>

ObjectManager::ObjectManager(QWidget* parent) :
		QWidget(parent) {

	QVBoxLayout * mainLayout01 = new QVBoxLayout(this);
	lineedit_search = new QLineEdit;
	QHBoxLayout * mainLayout02 = new QHBoxLayout;
	listwidgetList = new QListWidget;
	listwidgetList->setSelectionMode(QAbstractItemView::MultiSelection);
	QVBoxLayout * mainLayout03 = new QVBoxLayout;
	pushbutton_add = new QPushButton(">>");
	pushbutton_sub = new QPushButton("<<");
	mainLayout03->addWidget(pushbutton_add);
	mainLayout03->addWidget(pushbutton_sub);
	listwidgetBasket = new QListWidget;
	listwidgetBasket->setSelectionMode(QAbstractItemView::MultiSelection);
	mainLayout02->addWidget(listwidgetList);
	mainLayout02->addLayout(mainLayout03);
	mainLayout02->addWidget(listwidgetBasket);
	pushbutton_databaseUpdate = new QPushButton("DataBase Update");

	labelSearchHelp = new QLabel("");
	labelSearchHelp->setVisible(false);

	mainLayout01->addWidget(labelSearchHelp);
	mainLayout01->addWidget(lineedit_search);
	mainLayout01->addLayout(mainLayout02);
	mainLayout01->addWidget(pushbutton_databaseUpdate);

	setContextMenu(true);

	connect(lineedit_search, SIGNAL(textChanged(QString)), this, SLOT(trt_SearchChange(QString)));
    connect(listwidgetBasket, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_basketListClick(QListWidgetItem*)));
    connect(listwidgetBasket, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	connect(pushbutton_add, SIGNAL(clicked()), this, SLOT(trt_basketAdd()));
	connect(pushbutton_sub, SIGNAL(clicked()), this, SLOT(trt_basketSub()));
	connect(pushbutton_databaseUpdate, SIGNAL(clicked()), this, SLOT(trt_dataBaseUpdate()));

	connect(listwidgetList, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(ProvideContextMenuList(const QPoint &)));
	connect(listwidgetBasket, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(ProvideContextMenubasket(const QPoint &)));

	GlobalConfig& config = GlobalConfig::getConfig();
	setDatabasePath(config.databasePath());
	setVisibleBasket(false);
}


ObjectManager::~ObjectManager()
{


}

void ObjectManager::setVisibleBasket(bool val)
{
	visibleBasket = val;
	visibleBasketDisplay(visibleBasket);

}


void ObjectManager::setLabelSearchVisible(bool val)
{
	labelSearchHelp->setVisible(val);
}

void ObjectManager::setLabelSearchText(QString txt)
{
	labelSearchHelp->setText(txt);
}

void ObjectManager::setButtonDataBase(bool val)
{
	pushbutton_databaseUpdate->setVisible(val);
}


void ObjectManager::setListMultiSelection(bool type)
{
	if ( type )
	{
		listwidgetList->setSelectionMode(QAbstractItemView::MultiSelection);
	}
	else
	{
		listwidgetList->setSelectionMode(QAbstractItemView::SingleSelection);
	}
}

void ObjectManager::setListBasketMultiSelection(QAbstractItemView::SelectionMode type)
{
	listwidgetBasket->setSelectionMode(type);
}

void ObjectManager::setProjectType(int type)
{
	m_projectType = type;
}


void ObjectManager::setProjectName(QString name)
{
	m_projectName = name;
}

void ObjectManager::setSurveyName(QString name)
{
	m_surveyName = name;
}

void ObjectManager::setProjectCustomPath(QString path)
{
	m_projectCustomPath = path;
}

void ObjectManager::setContextMenu(bool val)
{
	if ( val )
	{
		listwidgetList->setContextMenuPolicy(Qt::CustomContextMenu);
		listwidgetBasket->setContextMenuPolicy(Qt::CustomContextMenu);
	}
	else
	{
		listwidgetList->setContextMenuPolicy(Qt::NoContextMenu);
		listwidgetBasket->setContextMenuPolicy(Qt::NoContextMenu);
	}
}

void ObjectManager::visibleBasketDisplay(bool val)
{
	listwidgetBasket->setVisible(val);
	pushbutton_add->setVisible(val);
	pushbutton_sub->setVisible(val);
}

int ObjectManager::getProjectType()
{
	return m_projectType;
}

QString ObjectManager::getProjectName()
{
	return m_projectName;
}

QString ObjectManager::getSurveyName()
{
	return m_surveyName;
}

QString ObjectManager::getProjectCustomPath()
{
	return m_projectCustomPath;
}

QList<QListWidgetItem*> ObjectManager::getBasketSelectedItems()
{
	return listwidgetBasket->selectedItems();
}

QString ObjectManager::getBasketSelectedName()
{
	QList<QListWidgetItem*> list0 = listwidgetBasket->selectedItems();
	if ( list0.empty() ) return "";
	return list0[0]->text();
}



void ObjectManager::clearBasket()
{
	m_namesBasket.clear();
	displayNamesBasket();
}

void ObjectManager::f_basketAdd()
{
	basketAdd();
	displayNamesBasket();
	listwidgetList->clearSelection();
	listwidgetBasket->clearSelection();
}

void ObjectManager::f_basketSub()
{
	basketSub();
	displayNamesBasket();
	listwidgetList->clearSelection();
	listwidgetBasket->clearSelection();
}

void ObjectManager::f_basketListClick(QListWidgetItem* listItem)
{

}

void ObjectManager::trt_SearchChange(QString str)
{
	displayNames();
}

void ObjectManager::trt_basketListClick(QListWidgetItem* listItem)
{
	f_basketListClick(listItem);
}

void ObjectManager::f_basketListSelectionChanged()
{

}

void ObjectManager::trt_basketListSelectionChanged()
{
	f_basketListSelectionChanged();
}

void ObjectManager::trt_basketAdd()
{
	f_basketAdd();
	/*
	basketAdd();
	displayNamesBasket();
	listwidgetList->clearSelection();
	listwidgetBasket->clearSelection();
	*/
}

void ObjectManager::trt_basketSub()
{
	f_basketSub();
	/*
	basketSub();
	displayNamesBasket();
	listwidgetList->clearSelection();
	listwidgetBasket->clearSelection();
	*/
}

void ObjectManager::setNames(ProjectManagerNames names)
{
	m_names = names;
}

void ObjectManager::setBasketNames(ProjectManagerNames names)
{
	m_namesBasket = names;
}

void ObjectManager::displayNames()
{
	listwidgetList->clear();
	std::vector<QString> full = m_names.getFull();
	std::vector<QString> tiny = m_names.getTiny();
	std::vector<QBrush> color = m_names.getColor();
	QString prefix = lineedit_search->text();
	for (int n=0; n<tiny.size(); n++)
	{
		QString str = tiny[n];
		if ( ProjectManagerNames::isMultiKeyInside(str, prefix ) )
		{
			QListWidgetItem *item = new QListWidgetItem;
			item->setText(str);
			item->setToolTip(str);
			item->setData(Qt::UserRole, full[n]);
			if ( color.size() > n )
				item->setForeground(color[n]);
			this->listwidgetList->addItem(item);
		}
	}
}

void ObjectManager::displayClear()
{
	listwidgetList->clear();
}

void ObjectManager::displayBasketClear()
{
	listwidgetBasket->clear();
}

void ObjectManager::dataClear()
{
	m_names.clear();
	displayClear();
}

void ObjectManager::dataBasketClear()
{
	m_namesBasket.clear();
	displayBasketClear();
}


void ObjectManager::displayNamesBasket()
{
	listwidgetBasket->clear();
	std::vector<QString> full = m_namesBasket.getFull();
	std::vector<QString> tiny = m_namesBasket.getTiny();
	std::vector<QBrush> color = m_namesBasket.getColor();

	for (int n=0; n<tiny.size(); n++)
	{
		QString str = tiny[n];
		QListWidgetItem *item = new QListWidgetItem;
		item->setText(str);
		item->setToolTip(str);
		item->setData(Qt::UserRole, full[n]);
		if ( color.size() > n )
			item->setForeground(color[n]);
		this->listwidgetBasket->addItem(item);
	}
}


void ObjectManager::basketAdd()
{
    QList<QListWidgetItem*> list0 = listwidgetList->selectedItems();
    std::vector<QString> tiny = m_names.getTiny();
    std::vector<QString> full = m_names.getFull();
    std::vector<QBrush> color = m_names.getColor();

    std::vector<QString> tinyBasket = m_namesBasket.getTiny();
    std::vector<QString> fullBasket = m_namesBasket.getFull();
    std::vector<QBrush> colorBasket = m_namesBasket.getColor();

    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = ProjectManagerNames::getIndexFromVectorString(tiny, txt);
        if ( idx >= 0 )
        {
        	// avoid duplicates
        	int idx_basket = ProjectManagerNames::getIndexFromVectorString(fullBasket, full[idx]);
        	if ( idx_basket < 0 )
        	{
				tinyBasket.push_back(tiny[idx]);
				fullBasket.push_back(full[idx]);
				if ( color.size() > idx )
				{
					colorBasket.push_back(color[idx]);
				}
        	}
        }
    }
    m_namesBasket.copy(tinyBasket, fullBasket, colorBasket);
}


void ObjectManager::basketSub()
{
	QList<QListWidgetItem*> list0 = listwidgetBasket->selectedItems();
	std::vector<QString> tinyBasket = m_namesBasket.getTiny();
	std::vector<QString> fullBasket = m_namesBasket.getFull();
	std::vector<QBrush> colorBasket = m_namesBasket.getColor();

    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = ProjectManagerNames::getIndexFromVectorString(tinyBasket, txt);
        if ( idx >= 0 )
        {
        	tinyBasket.erase(tinyBasket.begin()+idx, tinyBasket.begin()+idx+1);
        	fullBasket.erase(fullBasket.begin()+idx, fullBasket.begin()+idx+1);
            if ( colorBasket.size() > idx )
            	colorBasket.erase(colorBasket.begin()+idx, colorBasket.begin()+idx+1);
        }
    }
    m_namesBasket.copy(tinyBasket, fullBasket, colorBasket);
}




QString ObjectManager::getDatabasePath()
{
  // return QCoreApplication::applicationDirPath() + "/../DB/";
	return m_dataBasePath;
}

void ObjectManager::setDatabasePath(QString new_path)
{
	m_dataBasePath = new_path;
}


QString ObjectManager::getProjIndexNameForDataBase()
{
	if ( m_projectType >= 0 )
	{
		return QString::number(m_projectType);
	}
	else
	{
		QString pathTmp = m_projectCustomPath;
		pathTmp.replace("/", "_");
		QString ret = QString::number(5) + pathTmp ;
		return ret;
	}
}


void ObjectManager::f_dataBaseUpdate()
{

}

void ObjectManager::trt_dataBaseUpdate()
{
	f_dataBaseUpdate();
}


QString ObjectManager::getFullNamefromTinyName(QString tinyName)
{
	std::vector<QString> tiny = m_names.getTiny();
	std::vector<QString> full = m_names.getFull();
    int idx = ProjectManagerNames::getIndexFromVectorString(tiny, tinyName);
    if ( idx >= 0 )
    {
    	return full[idx];
    }
    return "";
}

void ObjectManager::ProvideContextMenu(QListWidget *listWidget, const QPoint &pos)
{
	QPoint item = listWidget->mapToGlobal(pos);
	QMenu submenu;
	submenu.addAction("info");
	submenu.addAction("folder");
	QAction* rightClickItem = submenu.exec(item);
	QListWidgetItem *item0 = listWidget->itemAt(pos);
	QString name = getFullNamefromTinyName(item0->text());
	QString path = ProjectManagerNames::getAbsolutePath(name);
	if (rightClickItem && rightClickItem->text().contains("info") )
	{

		FileInformationWidget dialog(name);
		int code = dialog.exec();
	}
	else if ( rightClickItem && rightClickItem->text().contains("folder") )
	{
		GlobalConfig& config = GlobalConfig::getConfig();
		QString cmd = config.fileExplorerProgram() + " " + path;
		system(cmd.toStdString().c_str());
	}
}


void ObjectManager::ProvideContextMenuList(const QPoint &pos)
{
	ProvideContextMenu(listwidgetList, pos);
}


void ObjectManager::ProvideContextMenubasket(const QPoint &pos)
{
	ProvideContextMenu(listwidgetBasket, pos);
}


QString ObjectManager::formatDirPath(const QString& path_to_format) {
	// to remove redundant /, . and ..
	QDir dir = QDir(path_to_format);
	QString absolute_path = dir.absolutePath();
	// add missing / at the end
	if (absolute_path.count()>0 && absolute_path.back()!='/')
	{
		absolute_path += "/";
	}
	return absolute_path;
}

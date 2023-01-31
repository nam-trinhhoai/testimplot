#include <QDebug>
#include <stdio.h>


#include <WellsHeadManager.h>
#include "wellbore.h"

WellsHeadManager::WellsHeadManager(QWidget* parent) :
		ObjectManager(parent) {

	setListBasketMultiSelection(QAbstractItemView::ExtendedSelection);
	setButtonDataBase(false);
	setLabelSearchVisible(true);
	setLabelSearchText("command line example log=sometext;tf2p=sometext;picks=sometext");
}

WellsHeadManager::~WellsHeadManager()
{

}



void WellsHeadManager::setForceDataBasket(WELLMASTER _data0)
{
	m_wellMaster = _data0;
	displayNamesBasket0();
	m_wellsLogManager->displayClear();
	m_wellsTF2PManager->displayClear();
	if ( m_wellsPicksManager )
		m_wellsPicksManager->displayClear();
	m_wellsLogManager->displayBasketClear();
	m_wellsTF2PManager->displayBasketClear();
	if ( m_wellsPicksManager )
		m_wellsPicksManager->displayBasketClear();
}


void WellsHeadManager::setWellPath(QString path)
{
	m_wellsPath = path;
	updateNames();
}

void WellsHeadManager::setWellsLogManager(WellsLogManager *wellsLogManager)
{
	m_wellsLogManager = wellsLogManager;
	m_wellsLogManager->setFinalWellbasket(&m_wellMaster.finalWellBasket);
}

void WellsHeadManager::setWellsTF2PManager(WellsTF2PManager *wellsTF2PManager)
{
	m_wellsTF2PManager = wellsTF2PManager;
	m_wellsTF2PManager->setFinalWellbasket(&m_wellMaster.finalWellBasket);
}

void WellsHeadManager::setWellsPicksManager(WellsPicksManager *wellsPicksManager)
{
	m_wellsPicksManager = wellsPicksManager;
	if ( m_wellsPicksManager )
		m_wellsPicksManager->setFinalWellbasket(&m_wellMaster.finalWellBasket);
}

void WellsHeadManager::setListData(std::vector<WELLHEADDATA> list)
{
	m_listData0 = list;
	setBasketNames(m_wellMaster.m_basketWellBore);
	displayNamesBasket();
}



void WellsHeadManager::displayNames()
{
	QString prefix = lineedit_search->text();
	listwidgetList->clear();
	for (int n1=0; n1<m_listData0.size(); n1++)
	{
		for (int n2=0; n2<m_listData0[n1].bore.size(); n2++)
		{
			if ( WellUtil::well_wellbore_display_valid(m_listData0, prefix, n1, n2)  )
			{
				// listwidgetList->addItem(m_listData0[n1].bore[n2].tinyName);
				QString name = m_listData0[n1].bore[n2].tinyName;
				QString path = m_listData0[n1].bore[n2].fullName;
				QListWidgetItem *item = new QListWidgetItem;
				item->setText(name);
				item->setToolTip(name);
				item->setData(Qt::UserRole, path);
				listwidgetList->addItem(item);
			}
		}
	}
}



void WellsHeadManager::displayNamesBasket0()
{
/*
	std::vector<QString> wellHead_tinyname_basket;
	std::vector<QString> wellHead_fullname_basket;
	std::vector<QString> wellBore_tinyname_basket;
	std::vector<QString> wellBore_fullname_basket;


	qDebug() << "******" << QString::number(m_wellMaster.finalWellBasket.size());

	for (int i=0; i<m_wellMaster.finalWellBasket.size(); i++)
	{
		wellHead_tinyname_basket.push_back(m_wellMaster.finalWellBasket[i].head_tinyname);
		wellHead_fullname_basket.push_back(m_wellMaster.finalWellBasket[i].head_fullname);
		qDebug() << m_wellMaster.finalWellBasket[i].head_tinyname;
		qDebug() << QString::number(m_wellMaster.finalWellBasket[i].wellborelist.size());
		for (int ii=0; ii<m_wellMaster.finalWellBasket[i].wellborelist.size(); ii++)
		{
			wellBore_tinyname_basket.push_back(m_wellMaster.finalWellBasket[i].wellborelist[ii].bore_tinyname);
			wellBore_fullname_basket.push_back(m_wellMaster.finalWellBasket[i].wellborelist[ii].bore_fullname);
		}
	}
	m_wellMaster.m_basketWell.copy(wellHead_tinyname_basket, wellHead_fullname_basket);
	m_wellMaster.m_basketWellBore.copy(wellBore_tinyname_basket, wellBore_fullname_basket);
	// listwidgetList->clearSelection();
	// setBasketNames(m_basketWellBore);
	setBasketNames(m_wellMaster.m_basketWell);
	displayNamesBasket();
	m_wellsLogManager->displayNames();
	m_wellsTF2PManager->displayNames();
	m_wellsPicksManager->displayNames();
	m_wellsLogManager->displayNamesBasket();
	m_wellsTF2PManager->displayNamesBasket();
	m_wellsPicksManager->displayNamesBasket();
	*/
	listwidgetList->clearSelection();
	setBasketNames(m_wellMaster.m_basketWellBore);
	displayNamesBasket();
}

void WellsHeadManager::f_basketAdd()
{
	// qDebug() << "ok";

	QList<QListWidgetItem*> list0 = listwidgetList->selectedItems();
	int idx_well, idx_bore;
	int N = list0.size();

	std::vector<QString> wellHead_tinyname_basket;
	std::vector<QString> wellHead_fullname_basket;
	std::vector<QString> wellBore_tinyname_basket;
	std::vector<QString> wellBore_fullname_basket;

	std::vector<QString> current_wellBore_fullname_basket = m_wellMaster.m_basketWellBore.getFull();
	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->data(Qt::UserRole).toString();
		WellUtil::getIndexFromWellBoreFullname(m_listData0, txt, &idx_well, &idx_bore);
		if ( idx_well >= 0 && idx_bore >= 0 )
		{
			// avoid duplicates
			int idx_basket = ProjectManagerNames::getIndexFromVectorString(current_wellBore_fullname_basket,
					m_listData0[idx_well].bore[idx_bore].fullName);
			if ( idx_basket < 0 ) {
				// well_wellbore_basket.push_back(display0.wells[idx_well].bore[idx_bore].bore_tinyname);
				wellHead_tinyname_basket.push_back(m_listData0[idx_well].tinyName);
				wellHead_fullname_basket.push_back(m_listData0[idx_well].fullName);
				wellBore_tinyname_basket.push_back(m_listData0[idx_well].bore[idx_bore].tinyName);
				wellBore_fullname_basket.push_back(m_listData0[idx_well].bore[idx_bore].fullName);
			}
		}
	}
	m_wellMaster.m_basketWell.add(wellHead_tinyname_basket, wellHead_fullname_basket);
	m_wellMaster.m_basketWellBore.add(wellBore_tinyname_basket, wellBore_fullname_basket);
	listwidgetList->clearSelection();
	setBasketNames(m_wellMaster.m_basketWellBore);
	displayNamesBasket();
	// qlw_welllog->clear();
	// qlw_welltf2p->clear();
	// qlw_wellpicks->clear();
	selectDefaultTFP(wellHead_tinyname_basket, wellHead_fullname_basket,
			wellBore_tinyname_basket, wellBore_fullname_basket);
}




void WellsHeadManager::f_basketSub()
{
	QList<QListWidgetItem*> list0 = listwidgetBasket->selectedItems();
	if ( list0.size() <= 0 ) return;

	std::vector<QString> wellHead_tinyname_basket;
	std::vector<QString> wellHead_fullname_basket;
	std::vector<QString> wellBore_tinyname_basket;
	std::vector<QString> wellBore_fullname_basket;

	wellHead_tinyname_basket = m_wellMaster.m_basketWell.getTiny();
	wellHead_fullname_basket = m_wellMaster.m_basketWell.getFull();
	wellBore_tinyname_basket = m_wellMaster.m_basketWellBore.getTiny();
	wellBore_fullname_basket = m_wellMaster.m_basketWellBore.getFull();

	for (int i=0; i<list0.size(); i++)
	{
		int idx_well = -1, idx_bore = -1;
		QString txt = list0[i]->text();
		int idx = ProjectManagerNames::getIndexFromVectorString(wellBore_tinyname_basket, txt);
		QString name = wellBore_fullname_basket[idx];
		WellUtil::getIndexFromWellBoreFullname(m_listData0, name, &idx_well, &idx_bore);

		// if ( idx_well >= 0 && idx_bore >= 0 )
		//	m_wellMaster.finalWellBasket[idx_well].wellborelist.erase(m_wellMaster.finalWellBasket[idx_well].wellborelist.begin()+idx_bore, m_wellMaster.finalWellBasket[idx_well].wellborelist.begin()+idx_bore+1);
	}

	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		// int idx = getIndexFromVectorString(well_wellbore_basket, txt);
		int idx = ProjectManagerNames::getIndexFromVectorString(wellBore_tinyname_basket, txt);
		if ( idx >= 0 )
		{
			// well_wellbore_basket.erase(well_wellbore_basket.begin()+idx, well_wellbore_basket.begin()+idx+1);
			wellHead_tinyname_basket.erase(wellHead_tinyname_basket.begin()+idx, wellHead_tinyname_basket.begin()+idx+1);
			wellHead_fullname_basket.erase(wellHead_fullname_basket.begin()+idx, wellHead_fullname_basket.begin()+idx+1);
			wellBore_tinyname_basket.erase(wellBore_tinyname_basket.begin()+idx, wellBore_tinyname_basket.begin()+idx+1);
			wellBore_fullname_basket.erase(wellBore_fullname_basket.begin()+idx, wellBore_fullname_basket.begin()+idx+1);


		}
	}

	m_wellMaster.m_basketWell.copy(wellHead_tinyname_basket, wellHead_fullname_basket);
	m_wellMaster.m_basketWellBore.copy(wellBore_tinyname_basket, wellBore_fullname_basket);
	listwidgetList->clearSelection();
	setBasketNames(m_wellMaster.m_basketWellBore);
	displayNamesBasket();
}

/*
void WellsHeadManager::displayNamesBasket()
{
	qDebug() << "display";

}
*/

void WellsHeadManager::f_basketListSelectionChanged()
{
	QList<QListWidgetItem*> selection = listwidgetBasket->selectedItems();//lw_wellsbasket->selectedItems();
	if ( selection.size()==1 )
	{
		QString well_tiny_name = selection.first()->text();
		f_basketListClick(well_tiny_name);
	}
	else
	{
		if ( m_wellsLogManager )
		{
			m_wellsLogManager->displayClear();
			m_wellsLogManager->displayBasketClear();
		}
		if ( m_wellsTF2PManager )
		{
			m_wellsTF2PManager->displayClear();
			m_wellsTF2PManager->displayBasketClear();
		}
		if ( m_wellsPicksManager )
		{
			m_wellsPicksManager->displayClear();
			m_wellsPicksManager->displayBasketClear();
		}
	}
}

//void WellsHeadManager::f_basketListClick(QListWidgetItem* listItem)
//{
//	QString name = listItem->text();
//	f_basketListClick(name);
//}


void WellsHeadManager::f_basketListClick(QString wellTinyName)
{
	qDebug() << "click";
	std::vector<QString> wellBoreTinyName = m_wellMaster.m_basketWellBore.getTiny();
	std::vector<QString> wellBoreFullName = m_wellMaster.m_basketWellBore.getFull();

	int idx = ProjectManagerNames::getIndexFromVectorString(wellBoreTinyName, wellTinyName);
	if ( idx == -1 ) return;
	QString bore_fullname = wellBoreFullName[idx];

	std::vector<QString> wellTinyName0 = m_wellMaster.m_basketWell.getTiny();
	std::vector<QString> wellFullName0 = m_wellMaster.m_basketWell.getFull();

	welllogNamesUpdate(bore_fullname);
	m_wellsLogManager->setWellHeadNames(wellTinyName0[idx], wellFullName0[idx]);
	m_wellsLogManager->setWellBoreNames(wellBoreTinyName[idx], wellBoreFullName[idx]);
	m_wellsTF2PManager->setWellHeadNames(wellTinyName0[idx], wellFullName0[idx]);
	m_wellsTF2PManager->setWellBoreNames(wellBoreTinyName[idx], wellBoreFullName[idx]);
	if ( m_wellsPicksManager )
	{
		m_wellsPicksManager->setWellHeadNames(wellTinyName0[idx], wellFullName0[idx]);
		m_wellsPicksManager->setWellBoreNames(wellBoreTinyName[idx], wellBoreFullName[idx]);
	}

	welllogNamesUpdate(bore_fullname);
	m_wellsLogManager->displayNamesBasket();
	m_wellsTF2PManager->displayNamesBasket();
	if ( m_wellsPicksManager )
		m_wellsPicksManager->displayNamesBasket();

	qDebug() << wellBoreTinyName[idx] << wellBoreFullName[idx];
	qDebug() << wellTinyName0[idx] << wellFullName0[idx];
}



void WellsHeadManager::welllogNamesUpdate(QString path)
{
	qDebug() << "path: " + path;
	int idx_well = -1, idx_bore = -1;
	WellUtil::getIndexFromWellBoreFullname(m_listData0, path, &idx_well, &idx_bore);

	ProjectManagerNames logs = m_listData0[idx_well].bore[idx_bore].logs;
	ProjectManagerNames tf2p = m_listData0[idx_well].bore[idx_bore].tf2p;
	ProjectManagerNames picks = m_listData0[idx_well].bore[idx_bore].picks;

	m_wellsLogManager->setNames(logs);
	m_wellsLogManager->displayNames();

	m_wellsTF2PManager->setNames(tf2p);
	m_wellsTF2PManager->displayNames();

	if ( m_wellsPicksManager )
	{
		m_wellsPicksManager->setNames(picks);
		m_wellsPicksManager->displayNames();
	}
}


int WellsHeadManager::getBasketSelectedIndex()
{
	QList<QListWidgetItem*> list0 = listwidgetBasket->selectedItems();
	if ( list0.size()!=1 ) return -1;
	QString name = list0[0]->text();
	// std::vector<QString> basketTinyNames =
	for ( int i=0; i<m_wellMaster.m_basketWellBore.tiny.size(); i++)
	{
		if ( name.compare(m_wellMaster.m_basketWellBore.tiny[i] ) == 0 ) return i;
	}
	return -1;
}

QString WellsHeadManager::getBasketSelectedName()
{
	QList<QListWidgetItem*> list0 = getBasketSelectedItems();
	if ( list0.empty() ) return "";
	return list0[0]->text();
}


/*
void WellsHeadManager::display_wells_list(QString prefix)
{
	this->lw_wells->clear();

	for (int i_well=0; i_well<display0.wells.size(); i_well++)
	{
		for (int i_bore=0; i_bore<display0.wells[i_well].bore.size(); i_bore++)
		{

			if ( well_wellbore_display_valid(prefix, i_well, i_bore) )
			{
				QString name = display0.wells[i_well].bore[i_bore].bore_tinyname;
				QListWidgetItem *item = new QListWidgetItem;
				item->setText(name);
				item->setToolTip(name);
				this->lw_wells->addItem(item);

			}
		}
	}
}
*/


void WellsHeadManager::updateNames()
{
	/*
	QString path = m_wellsPath;
	QFileInfoList list = ProjectManagerNames::getDirectoryList(path);
	int N = list.size();
	names0.resize(N);
	for (int n_well=0; n_well<N; n_well++)
	{
		QFileInfo fileInfo = list[n_well];
	    QString filetinyname = fileInfo.fileName();
	    QString filefullname = fileInfo.absoluteFilePath();
	    names0[n_well].tiny = filetinyname;
	    names0[n_well].full = filefullname;

	    QFileInfoList list_bore = ProjectManagerNames::getDirectoryList(filefullname);
	    int Nbores = list_bore.size();





	    display0.wells[n_well].bore.resize(Nbores);

	    	for (int n_bore=0; n_bore<Nbores; n_bore++)
	    	{
	    		QFileInfo bore_fileInfo = list_bore[n_bore];
	    		QString bore_filetinyname = bore_fileInfo.fileName();
	    		QString bore_filefullname = bore_fileInfo.absoluteFilePath();
	    		display0.wells[n_well].bore[n_bore].bore_tinyname = QString("[ ") + filetinyname +QString(" ] ") + bore_filetinyname;
	    		display0.wells[n_well].bore[n_bore].bore_fullname = bore_filefullname;

	    		welldeviation_names_update(bore_filefullname, n_well, n_bore);
	    		welllog_names_update(bore_filefullname, n_well, n_bore);
	    		welltf2p_names_update(bore_filefullname, n_well, n_bore);
	    		wellpicks_names_update(bore_filefullname, n_well, n_bore);
	    	}
	    	fprintf(stderr, "%d %d\n", n_well, N);
	    }
	}
	*/
}

void WellsHeadManager::selectDefaultTFP(const std::vector<QString>& wellHead_tinyname_basket,
		const std::vector<QString>& wellHead_fullname_basket,
		const std::vector<QString>& wellBore_tinyname_basket,
		const std::vector<QString>& wellBore_fullname_basket) {
	for (int i=0; i<wellHead_tinyname_basket.size(); i++) {
		// select default tfp
		QString wellHeadTinyName = wellHead_tinyname_basket[i];
		QString wellHeadFullName = wellHead_fullname_basket[i];
		QString wellBoreTinyName = wellBore_tinyname_basket[i];
		QString wellBoreFullName = wellBore_fullname_basket[i];
		QString deviationfullname = wellBoreFullName + "/deviation";
		int idx_well = -1, idx_bore = -1;
		WellUtil::wellListCreateGetIndex(&(m_wellMaster.finalWellBasket), wellHeadTinyName, wellHeadFullName, wellBoreTinyName, wellBoreFullName, deviationfullname, &idx_well, &idx_bore);
		if ( idx_well < 0 || idx_bore < 0 ) continue;

		QDir wellBoreDir(wellBoreFullName);
		QStringList descFiles = wellBoreDir.entryList(QStringList() << "*.desc", QDir::Files);
		QString wellBoreDescFile;

		QString descWellBoreFile;
		QString tfpWellBoreFile;
		if (descFiles.size()>0) {
			wellBoreDescFile = descFiles[0];
			descWellBoreFile = wellBoreDir.absoluteFilePath(wellBoreDescFile);
			tfpWellBoreFile = WellBore::getTfpFileFromDescFile(descWellBoreFile);
		}
		if (!tfpWellBoreFile.isNull() && !tfpWellBoreFile.isEmpty() && QFileInfo(tfpWellBoreFile).exists()) {
			QString tfpWellBoreName = ProjectManagerNames::getKeyTabFromFilename(tfpWellBoreFile, "Name");
			m_wellMaster.finalWellBasket[idx_well].wellborelist[idx_bore].tf2p_tinyname.clear();
			m_wellMaster.finalWellBasket[idx_well].wellborelist[idx_bore].tf2p_fullname.clear();
			m_wellMaster.finalWellBasket[idx_well].wellborelist[idx_bore].tf2p_displayname.clear();
			m_wellMaster.finalWellBasket[idx_well].wellborelist[idx_bore].tf2p_tinyname.push_back(tfpWellBoreName);
			m_wellMaster.finalWellBasket[idx_well].wellborelist[idx_bore].tf2p_fullname.push_back(tfpWellBoreFile);
			m_wellMaster.finalWellBasket[idx_well].wellborelist[idx_bore].tf2p_displayname.push_back(tfpWellBoreName);
		}
	}
}


std::vector<std::vector<QString>> WellsHeadManager::getWellBasketLogPicksTf2pNames(QString type, QString nameType)
{
	std::vector<std::vector<QString>> vout;

	std::vector<QString> wellBoreTinyName = m_wellMaster.m_basketWellBore.getTiny();
	std::vector<QString> wellBoreFullName = m_wellMaster.m_basketWellBore.getFull();

	vout.resize(wellBoreTinyName.size());
	for (int i=0; i<wellBoreTinyName.size(); i++)
	{
		int idx = i;
		if ( idx == -1 ) return vout;
		QString bore_fullname = wellBoreFullName[idx];
		// qDebug() << ">>>>  " + wellBoreFullName[idx];

		std::vector<QString> wellTinyName0 = m_wellMaster.m_basketWell.getTiny();
		std::vector<QString> wellFullName0 = m_wellMaster.m_basketWell.getFull();

		int idx_well = -1, idx_bore = -1;
		WellUtil::getIndexFromWellBoreFullname(m_listData0, bore_fullname, &idx_well, &idx_bore);

		ProjectManagerNames projectType;

		if ( type.compare("log") == 0 )
		{
			projectType = m_listData0[idx_well].bore[idx_bore].logs;
		}
		else if ( type.compare("picks") == 0 )
		{
			projectType = m_listData0[idx_well].bore[idx_bore].picks;
		}
		else if ( type.compare("tf2p") == 0 )
		{
			projectType = m_listData0[idx_well].bore[idx_bore].tf2p;
		}
		else
		{
			fprintf(stderr, "getWellBasketLogPicksTf2pNames unknown nameType: %s\n", nameType.toStdString().c_str());
			return vout;
		}

		std::vector<QString> names;
		if ( nameType.compare("tiny") == 0 ) names = projectType.getTiny();
		else if ( nameType.compare("full") == 0 ) names = projectType.getFull();
		else
		{
			fprintf(stderr, "getWellBasketLogPicksTf2pNames unknown type: %s\n", type.toStdString().c_str());
		}
		for (int j=0; j<names.size(); j++)
		{
			vout[i].push_back(names[j]);
		}
	}
	return vout;
}


std::vector<QString> WellsHeadManager::getWellBasketTinyNames()
{
	return m_wellMaster.m_basketWellBore.getTiny();
}

std::vector<QString> WellsHeadManager::getWellBasketFullNames()
{
	return m_wellMaster.m_basketWellBore.getFull();
}



#include <QDebug>
#include <stdio.h>

#include <WellsManager.h>
#include <WellsPicksManager.h>

WellsPicksManager::WellsPicksManager(QWidget* parent) :
		ObjectManager(parent) {
	setButtonDataBase(false);

}

WellsPicksManager::~WellsPicksManager()
{

}



void WellsPicksManager::displayNamesBasket()
{
	listwidgetBasket->clear();

	QString deviationfullname = wellBoreFullName + "/deviation";

	int idx_well = -1, idx_bore = -1;
	WellUtil::wellListCreateGetIndex(m_finalWellBasket, wellHeadTinyName, wellHeadFullName, wellBoreTinyName, wellBoreFullName, deviationfullname, &idx_well, &idx_bore);
	if ( idx_well < 0 || idx_bore < 0 ) return;
	for (int j=0; j<(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_displayname.size(); j++)
	{
		QListWidgetItem *item = new QListWidgetItem;
		item->setText((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_displayname[j]);
		item->setToolTip((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_displayname[j]);
		listwidgetBasket->addItem(item);
	}
}

void WellsPicksManager::setWellManager(WellsManager *wellManager)
{
	m_wellManager = wellManager;
}


void WellsPicksManager::f_basketAdd()
{
	QString deviationfullname = wellBoreFullName + "/deviation";
	// qDebug() << deviationfullname;

	QList<QListWidgetItem*> list0 = listwidgetList->selectedItems();
	if ( list0.size() == 0 ) return;

	int idx_well = -1, idx_bore = -1;
	WellUtil::wellListCreateGetIndex(m_finalWellBasket, wellHeadTinyName, wellHeadFullName, wellBoreTinyName, wellBoreFullName, deviationfullname, &idx_well, &idx_bore);

	std::vector<QString> wellslog_tinyname = m_names.getTiny();
	std::vector<QString> wellslog_fullname = m_names.getFull();

	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		int idx = ProjectManagerNames::getIndexFromVectorString(wellslog_tinyname, txt);
		if ( idx >= 0 )
		{
			int idx_basket = ProjectManagerNames::getIndexFromVectorString((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_fullname, wellslog_fullname[idx]);
			if ( idx_basket < 0 )
			{
				(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_tinyname.push_back(wellslog_tinyname[idx]);
				(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_fullname.push_back(wellslog_fullname[idx]);
				QString string = wellslog_tinyname[idx];
				(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_displayname.push_back(string);
			}
		}
	}
	listwidgetList->clearSelection();
	displayNamesBasket();
}

void WellsPicksManager::f_basketSub()
{
	QList<QListWidgetItem*> list0 = listwidgetBasket->selectedItems();
	if ( list0.size() == 0 ) return;


	 for (int i=0; i<list0.size(); i++)
	 {
		 QString txt = list0[i]->text();
		 int idx_well = -1;
		 int idx_bore = -1;
		 int  idx = -1;
		 getIndexFromPicksName(txt, &idx_well, &idx_bore, &idx);
	     if ( idx_well >= 0 && idx_bore >= 0 )
	     {
	    	 // int idx = getIndexFromVectorString(well_list[idx_well].wellborelist[idx_bore].log_displayname, txt);
	    	 (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_tinyname.erase((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_tinyname.begin()+idx, (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_tinyname.begin()+idx+1);
	    	 (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_fullname.erase((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_fullname.begin()+idx, (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_fullname.begin()+idx+1);
	    	 (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_displayname.erase((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_displayname.begin()+idx, (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].picks_displayname.begin()+idx+1);
	     }
	 }
	 listwidgetBasket->clearSelection();
	 displayNamesBasket();
}



void WellsPicksManager::setFinalWellbasket(std::vector<WELLLIST> *data)
{
	m_finalWellBasket = data;
}


void WellsPicksManager::setWellHeadBasket(ProjectManagerNames *names)
{
	m_wellHeadBasket = names;
}

void WellsPicksManager::setWellBoreHeadBasket(ProjectManagerNames *names)
{
	m_wellBoreHeadBasket = names;
}

void WellsPicksManager::setWellHeadNames(QString tiny, QString full)
{
	wellHeadTinyName = tiny;
	wellHeadFullName = full;
}

void WellsPicksManager::setWellBoreNames(QString tiny, QString full)
{
	wellBoreTinyName = tiny;
	wellBoreFullName = full;
}


void WellsPicksManager::getIndexFromPicksName(QString picks_displayname, int *idx_well, int *idx_bore, int *idx)
{
	*idx_well = -1;
	*idx_bore = -1;
	*idx = -1;

	QString name = m_wellManager->getWellsHeadBasketSelectedName();
	if ( name.isEmpty() ) return;

	for (int n=0; n<m_finalWellBasket->size(); n++)
	{
		for (int m=0; m<(*m_finalWellBasket)[n].wellborelist.size(); m++)
		{
			if ( name.compare((*m_finalWellBasket)[n].wellborelist[m].bore_tinyname) == 0 )
			{
				for (int p=0; p<(*m_finalWellBasket)[n].wellborelist[m].picks_displayname.size(); p++)
				{
					if ( (*m_finalWellBasket)[n].wellborelist[m].picks_tinyname[p].compare(picks_displayname) == 0 )
					{
						*idx_well = n;
						*idx_bore = m;
						*idx = p;
						return;
					}
				}
			}
		}
	}
}

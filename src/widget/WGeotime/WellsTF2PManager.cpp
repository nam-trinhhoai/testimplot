#include <QDebug>
#include <stdio.h>

#include <WellsManager.h>
#include <WellsTF2PManager.h>

WellsTF2PManager::WellsTF2PManager(QWidget* parent) :
		ObjectManager(parent) {
	setButtonDataBase(false);

}

WellsTF2PManager::~WellsTF2PManager()
{

}



void WellsTF2PManager::displayNamesBasket()
{
	listwidgetBasket->clear();

	QString deviationfullname = wellBoreFullName + "/deviation";

	int idx_well = -1, idx_bore = -1;
	WellUtil::wellListCreateGetIndex(m_finalWellBasket, wellHeadTinyName, wellHeadFullName, wellBoreTinyName, wellBoreFullName, deviationfullname, &idx_well, &idx_bore);
	if ( idx_well < 0 || idx_bore < 0 ) return;
	for (int j=0; j<(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname.size(); j++)
	{
		QListWidgetItem *item = new QListWidgetItem;
		item->setText((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname[j]);
		item->setToolTip((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname[j]);
		listwidgetBasket->addItem(item);
	}
}

void WellsTF2PManager::setWellManager(WellsManager *wellManager)
{
	m_wellManager = wellManager;
}


void WellsTF2PManager::f_basketAdd()
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
			// avoid duplicates
			int idx_basket = ProjectManagerNames::getIndexFromVectorString(
					(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_fullname, wellslog_fullname[idx]);
			if ( idx_basket < 0 ) {
				(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_tinyname.push_back(wellslog_tinyname[idx]);
				(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_fullname.push_back(wellslog_fullname[idx]);
				QString string = wellslog_tinyname[idx];
				(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname.push_back(string);
			}
		}
	}
	listwidgetList->clearSelection();
	displayNamesBasket();
}

void WellsTF2PManager::f_basketSub()
{
	QList<QListWidgetItem*> list0 = listwidgetBasket->selectedItems();
	if ( list0.size() == 0 ) return;


	 for (int i=0; i<list0.size(); i++)
	 {
		 QString txt = list0[i]->text();
		 int idx_well = -1;
		 int idx_bore = -1;
		 int  idx = -1;
		 getIndexFromTF2PName(txt, &idx_well, &idx_bore, &idx);
	     if ( idx_well >= 0 && idx_bore >= 0 )
	     {
	    	 // int idx = getIndexFromVectorString(well_list[idx_well].wellborelist[idx_bore].log_displayname, txt);
	    	 (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_tinyname.erase((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_tinyname.begin()+idx, (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_tinyname.begin()+idx+1);
	    	 (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_fullname.erase((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_fullname.begin()+idx, (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_fullname.begin()+idx+1);
	    	 (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname.erase((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname.begin()+idx, (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname.begin()+idx+1);
	     }
	 }
	 listwidgetBasket->clearSelection();
	 displayNamesBasket();
}



void WellsTF2PManager::setFinalWellbasket(std::vector<WELLLIST> *data)
{
	m_finalWellBasket = data;
}


void WellsTF2PManager::setWellHeadBasket(ProjectManagerNames *names)
{
	m_wellHeadBasket = names;
}

void WellsTF2PManager::setWellBoreHeadBasket(ProjectManagerNames *names)
{
	m_wellBoreHeadBasket = names;
}

void WellsTF2PManager::setWellHeadNames(QString tiny, QString full)
{
	wellHeadTinyName = tiny;
	wellHeadFullName = full;
}

void WellsTF2PManager::setWellBoreNames(QString tiny, QString full)
{
	wellBoreTinyName = tiny;
	wellBoreFullName = full;
}


void WellsTF2PManager::getIndexFromTF2PName(QString tf2p_displayname, int *idx_well, int *idx_bore, int *idx)
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
				for (int p=0; p<(*m_finalWellBasket)[n].wellborelist[m].tf2p_displayname.size(); p++)
				{
					if ( (*m_finalWellBasket)[n].wellborelist[m].tf2p_tinyname[p].compare(tf2p_displayname) == 0 )
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

void WellsTF2PManager::clearBasket() {
	if ( listwidgetBasket->count() == 0 ) return;

	for (int i=listwidgetBasket->count()-1; i>=0; i--)
	{
		QString txt = listwidgetBasket->item(i)->text();
		int idx_well = -1;
		int idx_bore = -1;
		int  idx = -1;
		getIndexFromTF2PName(txt, &idx_well, &idx_bore, &idx);
		if ( idx_well >= 0 && idx_bore >= 0 )
		{
			// int idx = getIndexFromVectorString(well_list[idx_well].wellborelist[idx_bore].log_displayname, txt);
			(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_tinyname.erase((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_tinyname.begin()+idx, (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_tinyname.begin()+idx+1);
			(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_fullname.erase((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_fullname.begin()+idx, (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_fullname.begin()+idx+1);
			(*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname.erase((*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname.begin()+idx, (*m_finalWellBasket)[idx_well].wellborelist[idx_bore].tf2p_displayname.begin()+idx+1);
		}
	}
	listwidgetBasket->clearSelection();
	displayNamesBasket();
}

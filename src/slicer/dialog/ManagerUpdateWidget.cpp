#include "ManagerUpdateWidget.h"

#include <QPushButton>
#include <QComboBox>
#include <QRadioButton>
#include <QLineEdit>
#include <QToolButton>
#include <QSplitter>
#include <QVBoxLayout>
#include <QTreeWidgetItem>
#include <QLabel>
#include <QDebug>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <algorithm>
#include "wellbore.h"
#include "wellhead.h"
#include <workingsetmanager.h>
#include <folderdata.h>
#include "horizonfolderdata.h"
#include <freehorizon.h>
#include <freeHorizonQManager.h>
#include <seismicinformationaggregator.h>
#include <nextvisionhorizoninformationaggregator.h>
#include "wellinformationaggregator.h"
#include "isohorizoninformationaggregator.h"
#include "nurbinformationaggregator.h"
#include <managerFileSelectorWidget.h>
#include "pickinformationaggregator.h"
#include <globalUtil.h>

using namespace std;

ManagerUpdateWidget::ManagerUpdateWidget(QString dataName,WorkingSetManager *manager,QWidget* parent):QWidget(parent){

	n_name = dataName;
//	m_SelectedBore = "";
	m_manager = manager ;
    setAttribute(Qt::WA_DeleteOnClose);

	if(n_name == "Seismics"){
		manager->getManagerWidget()->seismic_database_update();
		dataSeismicGui(manager);
		return;
	}
	else if(n_name == "Wells"){
		manager->getManagerWidget()->well_database_update();
		dataWellsGui(manager);
		return;
	}
	else if(n_name == "Markers"){
		manager->getManagerWidget()->pick_database_update();
		dataPicksGui(manager);
		return;
	}
	else if ( n_name == FREE_HORIZON_LABEL )
	{
		dataNextVisionHorizonGui(manager);
		return;
	}
	else if ( n_name == NV_ISO_HORIZON_LABEL )
	{
		dataIsoHorizonGui(manager);
		return;
	}
	else if( n_name == "Nurbs")
	{
		dataNurbsGui(manager);
	}
}


void ManagerUpdateWidget::dataSeismicGui(WorkingSetManager *manager)
{
    SeismicInformationAggregator* aggregator = new SeismicInformationAggregator(manager);
    ManagerFileSelectorWidget *widget = new ManagerFileSelectorWidget(aggregator);
    int code = widget->exec();
}

void ManagerUpdateWidget::dataNextVisionHorizonGui(WorkingSetManager *manager)
{
	NextvisionHorizonInformationAggregator* aggregator = new NextvisionHorizonInformationAggregator(manager);
	ManagerFileSelectorWidget *widget = new ManagerFileSelectorWidget(aggregator);
    int code = widget->exec();
}

void ManagerUpdateWidget::dataIsoHorizonGui(WorkingSetManager *manager)
{
	IsoHorizonInformationAggregator* aggregator = new IsoHorizonInformationAggregator(manager);
	ManagerFileSelectorWidget *widget = new ManagerFileSelectorWidget(aggregator);
	int code = widget->exec();

}

void ManagerUpdateWidget::dataWellsGui(WorkingSetManager *manager)
{
	WellInformationAggregator* aggregator = new WellInformationAggregator(manager);
	ManagerFileSelectorWidget *widget = new ManagerFileSelectorWidget(aggregator);
	int code = widget->exec();
}

void ManagerUpdateWidget::dataNurbsGui(WorkingSetManager *manager)
{
	NurbInformationAggregator* aggregator = new NurbInformationAggregator(manager);
	ManagerFileSelectorWidget *widget = new ManagerFileSelectorWidget(aggregator);
	int code = widget->exec();
}

void ManagerUpdateWidget::dataPicksGui(WorkingSetManager *manager)
{
	PickInformationAggregator* aggregator = new PickInformationAggregator(manager);
	ManagerFileSelectorWidget *widget = new ManagerFileSelectorWidget(aggregator);
	int code = widget->exec();
}


void ManagerUpdateWidget::dataChanged(const QModelIndex &topLeft,const QModelIndex &bottomRight, const QVector<int> &roles)
{
	for (int i : roles){
		if (i == Qt::CheckStateRole){

			QVariant var = topLeft.data(Qt::DisplayRole);
			QString text = var.toString();

			if (topLeft.data(Qt::CheckStateRole).toBool()){
				select_data(text);
			}else{
				unselect_data(text);
			}
		}
	}
}

void ManagerUpdateWidget::trt_SearchChange(QString text){
	display_data_tree(text);
}

//void ManagerUpdateWidget::updateData(QString text){
//	m_DataFullname.clear();
//	m_DataTinyname.clear();
//
//	m_SelectedDataTinyname.clear();
//	m_SelectedDataFullname.clear();
//
//	int wellIdx,indexBore,indexPick;
//
//	m_SelectedBore = text;
//
//	for(wellIdx = 0;wellIdx < m_wellDisplayList.size() ;wellIdx++){
//
//		if((m_wellDisplayList[wellIdx].head_tinyname == text) || (text == "All Wells")){
//
//			for(indexBore=0;indexBore< m_wellDisplayList[wellIdx].bore.size();indexBore++){
//
//				if(text != "All Wells"){
//					m_SelectedBore = m_wellDisplayList[wellIdx].bore[indexBore].bore_tinyname;
//				}
//				for( indexPick=0;indexPick < m_wellDisplayList[wellIdx].bore[indexBore].picks_tinyname.size();indexPick++){
//
//					auto result = std::find (m_DataFullname.begin(),m_DataFullname.end(),m_wellDisplayList[wellIdx].bore[indexBore].picks_fullname[indexPick]);
//					if(result == m_DataFullname.end())
//						m_DataFullname.push_back(m_wellDisplayList[wellIdx].bore[indexBore].picks_fullname[indexPick]);
//
//					result = std::find (m_DataTinyname.begin(),m_DataTinyname.end(),m_wellDisplayList[wellIdx].bore[indexBore].picks_tinyname[indexPick]);
//					if(result == m_DataTinyname.end())
//					    m_DataTinyname.push_back(m_wellDisplayList[wellIdx].bore[indexBore].picks_tinyname[indexPick]);
//				}
//			}
//
//			if(text != "All Wells"){
//				break;
//			}
//		}
//	}
//
//	this->display_data_tree();
//}

void ManagerUpdateWidget::trace(){
//	qDebug() << " BORE ::::: " << m_SelectedBore;

	for(int i = 0; i < m_wellList.size(); i++){

		qDebug() << "head_tinyname " << m_wellList[i].head_tinyname;
		qDebug() << "head_fullname " << m_wellList[i].head_fullname;

		for(int j = 0 ;j < m_wellList[i].wellborelist.size();j++){

			qDebug() << "----------------bore_fullname      " << m_wellList[i].wellborelist[j].bore_fullname;
			qDebug() << "----------------bore_tinyname      " << m_wellList[i].wellborelist[j].bore_tinyname;
			qDebug() << "----------------deviation_fullname " << m_wellList[i].wellborelist[j].deviation_fullname;

//			for(int k=0;k < m_wellList[i].wellborelist[j].picks_fullname.size();k++){
//
//				qDebug() << "---------------------------picks_fullname " << m_wellList[i].wellborelist[j].picks_fullname[k];
//				qDebug() << "---------------------------picks_tinyname " << m_wellList[i].wellborelist[j].picks_tinyname[k];
//			}
		}
	}
}

//void ManagerUpdateWidget::updatePick(QString &rstrItem){
//
//	for(int indexWell = 0;indexWell < m_wellList.size();indexWell++){
//
//		for(int indexBore=0;indexBore< m_wellList[indexWell].wellborelist.size();indexBore++){
//
//			qDebug() << m_SelectedBore << m_wellList[indexWell].wellborelist[indexBore].bore_tinyname ;
//			if((m_wellList[indexWell].wellborelist[indexBore].bore_tinyname == m_SelectedBore)||(m_SelectedBore == "All Wells")){
//
//				m_wellList[indexWell].wellborelist[indexBore].picks_fullname = m_SelectedDataFullname;
//				m_wellList[indexWell].wellborelist[indexBore].picks_tinyname = m_SelectedDataTinyname;
//			}
//		}
//	}
//}

void ManagerUpdateWidget::unselect_data(QString &rstrItem){

	std::vector<QString>::iterator it =  std::find(m_SelectedDataTinyname.begin(),m_SelectedDataTinyname.end(),rstrItem);
	if (it != m_SelectedDataTinyname.end()){

		int index = std::distance(m_SelectedDataTinyname.begin(), it);
		m_SelectedDataTinyname.erase(it);
		m_SelectedDataFullname.erase(m_SelectedDataFullname.begin()+index, m_SelectedDataFullname.begin()+index+1);
	}

	if(n_name == "Markers"){
		deletePick(rstrItem);
	}
	else if(n_name == "Wells"){
		deleteWell(rstrItem);
	}
	else if(n_name == "Nurbs"){
		std::vector<QString>::iterator it =  std::find(m_selectedNurbsName.begin(),m_selectedNurbsName.end(),rstrItem);
		if (it != m_selectedNurbsName.end()){

			int index = std::distance(m_selectedNurbsName.begin(), it);
			m_selectedNurbsName.erase(it);
			m_selectedNurbsFullname.erase(m_selectedNurbsFullname.begin()+index, m_selectedNurbsFullname.begin()+index+1);
		}

	}
}


void ManagerUpdateWidget::deletePick(QString &rstrItem)
{
	int i=0;
	bool notFound = true;
	while(notFound && i<m_picksNames.size()){
		notFound = rstrItem != m_picksNames[i];
		if(notFound) {
			i++;
		} else {
			m_picksNames.erase(m_picksNames.begin()+i,m_picksNames.begin()+i+1);
			m_picksPaths.erase(m_picksPaths.begin()+i,m_picksPaths.begin()+i+1);
			m_picksColors.erase(m_picksColors.begin()+i,m_picksColors.begin()+i+1);
		}
	}
	//trace();
}

void ManagerUpdateWidget::deleteWell(QString &rstrItem)
{
	for(int i = 0;i<m_wellList.size();i++){

		for(int j=0;j< m_wellList[i].wellborelist.size();j++){
			if(rstrItem == m_wellList[i].wellborelist[j].bore_tinyname){
				m_wellList[i].wellborelist.erase(m_wellList[i].wellborelist.begin()+j,m_wellList[i].wellborelist.begin()+j+1);
				if(m_wellList[i].wellborelist.size() == 0){
					m_wellList.erase(m_wellList.begin()+i,m_wellList.begin()+i+1);
					return;
				}
			}
		}
	}
	//trace();
}

void ManagerUpdateWidget::addPick(QString &rstrItem) {
	int i = 0;
	bool notFound = true;
	while (notFound && i<m_allPicksNames.size()) {
		notFound = m_allPicksNames[i]!=rstrItem;
		if (notFound) {
			i++;
		}
	}
	if (!notFound) {
		int j = 0;
		bool notIn = true;
		while (notIn && j<m_picksNames.size()) {
			notIn = m_picksNames[j]!=m_allPicksNames[i];
			if (notIn) {
				j++;
			}
		}
		if (notIn) {
			m_picksNames.push_back(m_allPicksNames[i]);
			m_picksPaths.push_back(m_allPicksPaths[i]);
			m_picksColors.push_back(m_allPicksColors[i]);
		}
	}
}

void ManagerUpdateWidget::addWell(int wellIdx,int indexBore){

	WELLLIST well;

	well.head_fullname = m_wellDisplayList[wellIdx].head_fullname;
	well.head_tinyname = m_wellDisplayList[wellIdx].head_tinyname;

	WELLBORELIST borelist;
	borelist.bore_tinyname = m_wellDisplayList[wellIdx].bore[indexBore].bore_tinyname ;
	borelist.bore_fullname = m_wellDisplayList[wellIdx].bore[indexBore].bore_fullname ;
	if(m_wellDisplayList[wellIdx].bore[indexBore].deviation_fullname != ""){
		borelist.deviation_fullname = m_wellDisplayList[wellIdx].bore[indexBore].deviation_fullname;
	}else {
		QString fielName  = borelist.bore_fullname + "/deviation";
		if ( QFile::exists(fielName) )
			borelist.deviation_fullname =  fielName;
	}

	borelist.tf2p_fullname = m_wellDisplayList[wellIdx].bore[indexBore].tf2p_fullname ;
	borelist.tf2p_tinyname = m_wellDisplayList[wellIdx].bore[indexBore].tf2p_tinyname ;

	well.wellborelist.push_back(borelist);
	m_wellList.push_back(well);

	trace();
}

bool ManagerUpdateWidget::checkWellManager(QString &rstrItem)
{
	bool bFound = false;

	for(WellHead* pWell:m_manager->listWellHead()){
		if(pWell->name() == rstrItem){
			bFound = true;
			break;
		}
	}

	return bFound;
}

void ManagerUpdateWidget::updateWells(QString &rstrItem){

	int wellIdx,indexBore;

	for(wellIdx = 0;wellIdx < m_wellDisplayList.size();wellIdx++){

		for(indexBore=0;indexBore< m_wellDisplayList[wellIdx].bore.size();indexBore++){

			if(m_wellDisplayList[wellIdx].bore[indexBore].bore_tinyname == rstrItem){

				if(checkWellManager(m_wellDisplayList[wellIdx].head_tinyname) == false){
					addWell(wellIdx,indexBore);
				}
				return;
			}
		}
	}
}


void ManagerUpdateWidget::select_data(QString &rstrItem){

	std::vector<QString>::iterator it =  std::find(m_DataTinyname.begin(),m_DataTinyname.end(),rstrItem);

	if (it != m_DataTinyname.end()){

		int index = std::distance(m_DataTinyname.begin(), it);
		m_SelectedDataTinyname.push_back(rstrItem);
		m_SelectedDataFullname.push_back(m_DataFullname[index]);
	}


	if(n_name == "Markers"){
		addPick(rstrItem);
	}
	else if(n_name == "Wells"){
		updateWells(rstrItem);
	}
	else  if(n_name == "Nurbs"){
		std::vector<QString>::iterator it =  std::find(m_DataTinyname.begin(),m_DataTinyname.end(),rstrItem);

		if (it != m_DataTinyname.end()){

			int index = std::distance(m_DataTinyname.begin(), it);
			m_selectedNurbsName.push_back(rstrItem);
			m_selectedNurbsFullname.push_back(m_DataFullname[index]);
		}
	}
}


QStringList ManagerUpdateWidget::spitSearchItem(QString line)
{
	QStringList list;
	list = line.split(" ");
	list.removeAll(QString(""));
	return list;
}

bool ManagerUpdateWidget::isDiplayName(QString& name, QStringList& list){
	if ( list.size() == 0 ) return true;
	for (int i=0; i<list.size(); i++)
		if ( !name.contains(list[i], Qt::CaseInsensitive) )  return false;
	return true;
}

/*
bool ManagerUpdateWidget::isDiplayName(std::vector<QString>& names, QStringList& list){
	if ( list.size() == 0 ) return true;
	if ( names.size() == 0 ) return false;

	std::vector<bool> v;
	v.resize(names.size(), false);
	for (int i=0; i<names.size(); i++)
	{
		QString name = names[i];
		for (int n=0; n<list.size(); n++)
		{
			if ( name.contains(list[n], Qt::CaseInsensitive) )  v[i] = true;
<<<<<<< HEAD
		}
	}
	for (int i=0; i<v.size(); i++) if ( !v[i] ) return false;
	return true;
}
*/

/*
bool ManagerUpdateWidget::isDiplayName(std::vector<QString>& names, QStringList& list){
	if ( list.size() == 0 ) return true;
	if ( names.size() == 0 ) return false;

	std::vector<bool> v;
	v.resize(list.size(), false);
	for (int n=0; n<list.size(); n++)
	{
		for (int i=0; i<names.size(); i++)
		{
			QString name = names[i];
			if ( name.contains(list[n], Qt::CaseInsensitive) )
			{
				v[n] = true;
				break;
			}
		}
	}
	for (int i=0; i<v.size(); i++) if ( !v[i] ) return false;
	return true;
}
*/

bool ManagerUpdateWidget::isDiplayName(std::vector<QString>& names, QStringList& list){
	if ( list.size() == 0 ) return true;
	if ( names.size() == 0 ) return false;

	std::vector<bool> v;
	v.resize(list.size(), false);
	for (int n=0; n<list.size(); n++)
	{
		for (int i=0; i<names.size(); i++)
		{
			QString name = names[i];
			if ( name.contains(list[n], Qt::CaseInsensitive) )
			{
				v[n] = true;
				break;
			}
		}
	}
	for (int i=0; i<v.size(); i++) if ( !v[i] ) return false;
	return true;
}

void ManagerUpdateWidget::displayWellsDataTree(QString prefix){

	this->m_Data_SelectionTree->clear();
	int nbElement = 0;

	QString prefixName = pLineEditSearch->text();
    QString prefixLog =  pLineEditSearchWellLog->text();
    QString prefixTf2p = pLineEditSearchWellTf2p->text();
    QString prefixPicks = pLineEditSearchWellPicks->text();

    QStringList searchName = spitSearchItem(prefixName);
    QStringList searchLogName = spitSearchItem(prefixLog);
    QStringList searchTf2pName = spitSearchItem(prefixTf2p);
    QStringList searchPicksName = spitSearchItem(prefixPicks);

	for ( int n=0; n<m_wellDisplayList.size(); n++)
	{
		for (int m=0; m<m_wellDisplayList[n].bore.size(); m++)
		{
			QString name = m_wellDisplayList[n].bore[m].bore_tinyname;
			if ( !isDiplayName(name, searchName) ) continue;
			if ( !isDiplayName(m_wellDisplayList[n].bore[m].log_tinyname, searchLogName) ) continue;
			if ( !isDiplayName(m_wellDisplayList[n].bore[m].tf2p_tinyname, searchTf2pName) ) continue;
			if ( !isDiplayName(m_wellDisplayList[n].bore[m].picks_tinyname, searchPicksName) ) continue;

			QTreeWidgetItem *item = new QTreeWidgetItem;

			item->setData(0, Qt::DisplayRole, QVariant::fromValue(name));
			item->setData(0, Qt::CheckStateRole, QVariant::fromValue(false));

			item->setToolTip(0,name);

			this->m_Data_SelectionTree->insertTopLevelItem(nbElement,item);
			nbElement++;
		}
	}
}

void ManagerUpdateWidget::display_data_tree(QString prefix){
	if ( n_name == "Wells" )
	{
		displayWellsDataTree(prefix);
		return;
	}

	this->m_Data_SelectionTree->clear();

	int nbElement = 0;
	for (int i=0; i < m_DataTinyname.size(); i++){
		if((m_DataTinyname[i].contains(prefix, Qt::CaseInsensitive) == true )|| (prefix == ""))
		{

			/*
			if ( ( ( n_name == FREE_HORIZON_LABEL || n_name == NV_ISO_HORIZON_LABEL )  && !isNameExist(m_DataTinyname[i], m_horizonInTree) ) ||
					( n_name != FREE_HORIZON_LABEL && n_name != NV_ISO_HORIZON_LABEL ) )
					*/

			if ( n_name == FREE_HORIZON_LABEL || n_name == NV_ISO_HORIZON_LABEL )
			{
				if ( !isNameExist(m_DataTinyname[i], m_horizonInTree) || m_forceAllItems )
				{
					QTreeWidgetItem *item = new QTreeWidgetItem;
					if ( n_name == FREE_HORIZON_LABEL )
					{
						QIcon icon = FreeHorizonQManager::getHorizonIcon(m_DataFullname[i], 16);
						item->setIcon(0, icon);
					}
					item->setData(0, Qt::DisplayRole, QVariant::fromValue(m_DataTinyname[i]));
					item->setData(0, Qt::CheckStateRole, QVariant::fromValue(false));
					item->setToolTip(0,m_DataTinyname[i]);
					this->m_Data_SelectionTree->insertTopLevelItem(nbElement,item);
					nbElement++;
				}
			}
			else
			{
				QTreeWidgetItem *item = new QTreeWidgetItem;
				if ( n_name == "Seismics" )
				{
					QIcon icon = FreeHorizonQManager::getDataSetIcon(m_DataFullname[i]);
					item->setIcon(0, icon);
				}
				item->setData(0, Qt::DisplayRole, QVariant::fromValue(m_DataTinyname[i]));
				item->setData(0, Qt::CheckStateRole, QVariant::fromValue(false));
				item->setToolTip(0,m_DataTinyname[i]);
				this->m_Data_SelectionTree->insertTopLevelItem(nbElement,item);
				nbElement++;
			}
		}
	}

}

ManagerUpdateWidget::~ManagerUpdateWidget(){

}

std::vector<MARKER> ManagerUpdateWidget::getPicksList() {
	return m_manager->getManagerWidget()->staticGetPicksSortedWells(m_picksNames, m_picksColors, m_wellDisplayList);
}


std::vector<QString> ManagerUpdateWidget::getHorizonInTree()
{
	std::vector<QString> data;
	FolderData *folderData = m_manager->folders().horizonsFree;
	QList<IData*> list0 = folderData->data();
	data.resize(list0.size());
	for (int i=0; i<list0.size(); i++)
	{
		data[i] = list0[i]->name();
	}
	return data;
}


bool ManagerUpdateWidget::isNameExist(QString name, std::vector<QString> list)
{
	for ( QString name0:list )
		if ( name0 == name )
			return true;
	return false;
}

bool ManagerUpdateWidget::forceAllItems() const {
	return m_forceAllItems;
}

void ManagerUpdateWidget::setForceAllItems(bool val) {
	if (val!=m_forceAllItems) {
		m_forceAllItems = val;
		QString prefix;
		if (pLineEditSearch) {
			prefix = pLineEditSearch->text();
		}
		display_data_tree(prefix);
	}
}


#include "DataUpdatorDialog.h"

#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <QGridLayout>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QPushButton>

#include "DataSelectorDialog.h"
#include "ManagerUpdateWidget.h"
#include "workingsetmanager.h"
#include "wellhead.h"
#include "wellbore.h"
#include "wellpick.h"
#include "folderdata.h"
#include "globalconfig.h"
#include "marker.h"
#include "seismicsurvey.h"
#include <globalUtil.h>


DataUpdatorDialog::DataUpdatorDialog(QString dataName,WorkingSetManager *manager,QWidget *parent){
	m_manager = manager;
	QString title= dataName + " Update";
	setWindowTitle(title);
	m_DataName = dataName;
	setAttribute(Qt::WA_DeleteOnClose);
	m_Updator = new ManagerUpdateWidget(dataName,m_manager,this);
	accept();
}

DataUpdatorDialog::~DataUpdatorDialog() {

}

std::vector<QString> DataUpdatorDialog::getPathSelected()
{
	return m_Updator->getDataFullName();
}



void DataUpdatorDialog::accepted() {

	//qDebug()<<"start 1";
	//std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	QString surveyPath = m_manager->getManagerWidget()->get_survey_fullpath_name();
	QString surveyName = m_manager->getManagerWidget()->get_survey_name();

	std::vector<QString>  datasetPaths = m_Updator->getDataFullName();
	const std::vector<QString>&  datasetNames = m_Updator->getDataTinyName();
	const std::vector<WELLLIST>& wells = m_Updator->getWellList();
	const std::vector<MARKER>& picksList = m_Updator->getPicksList();

	qDebug() << surveyPath << surveyName;

	/*
	if ( m_DataName != FREE_HORIZON_LABEL && m_DataName != NV_ISO_HORIZON_LABEL )
		for(int i=0;i<datasetPaths.size();i++){
			datasetPaths[i] = m_manager->getManagerWidget()->get_seismic_path0() + datasetPaths[i];
		}
		*/

	if(m_DataName == "Seismics" ){
		bool bIsNewSurvey = false;
		SeismicSurvey* baseSurvey = DataSelectorDialog::dataGetBaseSurvey(m_manager, surveyName, surveyPath, bIsNewSurvey);
		if(baseSurvey != nullptr){
			DataSelectorDialog::createSeismic(baseSurvey,m_manager,datasetPaths,datasetNames,bIsNewSurvey,false);
		}
	}
	else if((m_DataName == "Markers")|| (m_DataName == "Wells")){
		DataSelectorDialog::addWellBore(m_manager,wells,picksList,false);
	}
	else if ( m_DataName == FREE_HORIZON_LABEL )
	{
		bool bIsNewSurvey = false;
		SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(m_manager, surveyName, surveyPath, bIsNewSurvey);
		DataSelectorDialog::addNVHorizons(m_manager, survey, datasetPaths, datasetNames);
	}
	else if ( m_DataName == NV_ISO_HORIZON_LABEL )
	{
		bool bIsNewSurvey = false;
		SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(m_manager, surveyName, surveyPath, bIsNewSurvey);
		DataSelectorDialog::addNVIsoHorizons(m_manager, survey, datasetPaths, datasetNames);
	}
	else if ( m_DataName == "Nurbs" )
	{
		std::vector<QString>  nurbsname = m_Updator->getSelectedNurbsName();
		std::vector<QString>  nurbspath = m_Updator->getSelectedNurbsFullname();
		DataSelectorDialog::addNurbs(m_manager,nurbspath, nurbsname);
	}

	accept();

//	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//	qDebug() << "finish ::show: " << std::chrono::duration<double, std::milli>(end-start).count();
}

bool DataUpdatorDialog::forceAllItems() const {
	return m_Updator->forceAllItems();
}

void DataUpdatorDialog::setForceAllItems(bool val) {
	m_Updator->setForceAllItems(val);
}



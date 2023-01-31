#include "globalUtil.h"
#include "horizonAttributComputeDialog.h"
#include "horizonfolderdatarep.h"
#include "horizonfolderdata.h"
#include "horizonproppanel.h"
#include "DataUpdatorDialog.h"
#include "abstractinnerview.h"
#include "importsismagehorizondialog.h"
//#include "computereflectivitywidget.h"

#include <QMenu>
#include <QAction>


HorizonFolderDataRep::HorizonFolderDataRep(HorizonFolderData *folderData, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = folderData;
	m_name = folderData->name();



}

HorizonFolderDataRep::~HorizonFolderDataRep() {

}

QWidget* HorizonFolderDataRep::propertyPanel() {

	/*if (m_propPanel == nullptr)
	{
		m_propPanel = new HorizonPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this](){
				m_propPanel = nullptr;
		});
	}
	return m_propPanel;*/
	return nullptr;


}

GraphicLayer * HorizonFolderDataRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

IData* HorizonFolderDataRep::data() const {
	return m_data;
}

//void HorizonFolderDataRep::buildContextMenu(QMenu *menu) {
	/*if (m_name == "St. horizons") {
			QAction *HorizonAction = new QAction("Animation horizon", this);
			menu->addAction(HorizonAction);
			connect(HorizonAction, SIGNAL(triggered()), this, SLOT(showHorizonWidget()));
		}*/


	/*if(m_name == "Wells" || m_name == "Markers"){
		QString strLabel = "Add " + m_name;
		QAction *addAction = new QAction(strLabel, this);
		menu->addAction(addAction);
		connect(addAction, SIGNAL(triggered()), this, SLOT(addData()));
	}
	if (m_name == "Wells") {
		QAction *computeReflectivityAction = new QAction("Compute Reflectivity", this);
		menu->addAction(computeReflectivityAction);
		connect(computeReflectivityAction, SIGNAL(triggered()), this, SLOT(computeReflectivity()));
	}*/

	/*if (m_name == FREE_HORIZON_LABEL ) {
		QAction *deleteNVHorizonAction = new QAction("delete Horizons", this);
		//QAction *addSismageHorizonAction = new QAction("Add Sismage Horizons", this);
		//QAction *computeFreeHorizonAttibutAction = new QAction("Compute attributs", this);
		menu->addAction(deleteNVHorizonAction);
		//menu->addAction(addSismageHorizonAction);
		//menu->addAction(computeFreeHorizonAttibutAction);
		connect(deleteNVHorizonAction, SIGNAL(triggered()), this, SLOT(addData()));
		//connect(addSismageHorizonAction, SIGNAL(triggered()), this, SLOT(addSismageHorizon()));
		//connect(computeFreeHorizonAttibutAction, SIGNAL(triggered()), this, SLOT(computeAttributHorizon()));
	}*/
//}
/*
void HorizonFolderDataRep::addData(){
	DataUpdatorDialog *dialog = new DataUpdatorDialog(m_name,m_data->workingSetManager(),nullptr);
	dialog->show();
}*/

void HorizonFolderDataRep::showHorizonWidget() {
	//ComputeReflectivityWidget* widget = new ComputeReflectivityWidget(m_data);
	//widget->show();
}
/*
void HorizonFolderDataRep::addSismageHorizon()
{
	ImportSismageHorizonDialog *p = new ImportSismageHorizonDialog(m_data->workingSetManager());
	p->show();
}


void HorizonFolderDataRep::computeAttributHorizon()
{
	HorizonAttributComputeDialog *p = new HorizonAttributComputeDialog(nullptr);
	p->setProjectManager(m_data->workingSetManager()->getManagerWidget());
	p->show();
}*/

AbstractGraphicRep::TypeRep HorizonFolderDataRep::getTypeGraphicRep() {
	return AbstractGraphicRep::NotDefined;
}

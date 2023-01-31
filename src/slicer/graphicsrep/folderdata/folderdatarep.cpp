#include "folderdatarep.h"
#include "folderdata.h"
#include "DataUpdatorDialog.h"
#include "DataSelectorDialog.h"
#include "computereflectivitywidget.h"
#include <globalUtil.h>
#include <importsismagehorizondialog.h>
#include <horizonAttributComputeDialog.h>
#include "ManagerUpdateWidget.h"
#include <workingsetmanager.h>
#include "horizonfolderdata.h"
#include "pickinformationaggregator.h"
#include "nurbinformationaggregator.h"

#include "seismicinformationaggregator.h"
#include "nextvisionhorizoninformationaggregator.h"
#include "isohorizoninformationaggregator.h"

#include "horizonanimaggregator.h"
#include "wellinformationaggregator.h"
#include "managerwidget.h"
#include <freeHorizonManager.h>
#include <QMenu>
#include <QAction>

FolderDataRep::FolderDataRep(FolderData *folderData, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = folderData;
	m_name = folderData->name();
}

FolderDataRep::~FolderDataRep() {

}

QWidget* FolderDataRep::propertyPanel() {
	return nullptr;
}

GraphicLayer * FolderDataRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

IData* FolderDataRep::data() const {
	return m_data;
}

void FolderDataRep::buildContextMenu(QMenu *menu) {
	if(m_name == "Wells" || m_name == "Markers"){
		QString strLabel = "Add " + m_name;
		QAction *addAction = new QAction(strLabel, this);
		menu->addAction(addAction);
		connect(addAction, SIGNAL(triggered()), this, SLOT(addData()));
	}
	if (m_name == "Wells") {
		QAction *informationAction = new QAction("Information", this);
		menu->addAction(informationAction);
		connect(informationAction, SIGNAL(triggered()), this, SLOT(openWellInformation()));

		QAction *computeReflectivityAction = new QAction("Compute Reflectivity", this);
		menu->addAction(computeReflectivityAction);
		connect(computeReflectivityAction, SIGNAL(triggered()), this, SLOT(computeReflectivity()));
	}
	if (m_name == "Markers") {
		QAction *informationAction = new QAction("Information", this);
		menu->addAction(informationAction);
		connect(informationAction, SIGNAL(triggered()), this, SLOT(openPicksInformation()));
	}
	if (m_name == FREE_HORIZON_LABEL ) {
		QAction *infoNVHorizonAction = new QAction("Information", this);
		QAction *addNVHorizonAction = new QAction("Add Nextvision Horizons", this);
		QAction *addSismageHorizonAction = new QAction("Add Sismage Horizons", this);
		QAction *computeFreeHorizonAttibutAction = new QAction("Compute attributs", this);
		//QAction *FreeHorizonLoadAnimationAction = new QAction("Load Animation", this);
		//QAction *FreeHorizonAnimationAction = new QAction("Play Animation", this);
		menu->addAction(infoNVHorizonAction);
		menu->addAction(addNVHorizonAction);
		menu->addAction(addSismageHorizonAction);
		menu->addAction(computeFreeHorizonAttibutAction);
		//menu->addAction(FreeHorizonLoadAnimationAction);
		//menu->addAction(FreeHorizonAnimationAction);
		connect(infoNVHorizonAction, SIGNAL(triggered()), this, SLOT(openNVHorizonInformation()));
		connect(addNVHorizonAction, SIGNAL(triggered()), this, SLOT(addData()));
		connect(addSismageHorizonAction, SIGNAL(triggered()), this, SLOT(addSismageHorizon()));
		connect(computeFreeHorizonAttibutAction, SIGNAL(triggered()), this, SLOT(computeAttributHorizon()));
		//connect(FreeHorizonAnimationAction, SIGNAL(triggered()), this, SLOT(playAnimation()));
		//connect(FreeHorizonLoadAnimationAction, SIGNAL(triggered()), this, SLOT(loadAnimation()));
	}
	if ( m_name == NV_ISO_HORIZON_LABEL )
	{
		QAction *infoNVIsoValAction = new QAction("Information", this);
		QAction *addNVIsoValAction = new QAction("Add Nextvision Isovalue Horizons", this);
		// QAction *computeFreeHorizonAttibutAction = new QAction("Compute attributs", this);
		menu->addAction(infoNVIsoValAction);
		menu->addAction(addNVIsoValAction);
		// menu->addAction(addSismageHorizonAction);
		// menu->addAction(computeFreeHorizonAttibutAction);
		connect(infoNVIsoValAction, SIGNAL(triggered()), this, SLOT(openIsoHorizonInformation()));
		connect(addNVIsoValAction, SIGNAL(triggered()), this, SLOT(addData()));
		// connect(addSismageHorizonAction, SIGNAL(triggered()), this, SLOT(addSismageHorizon()));
		// connect(computeFreeHorizonAttibutAction, SIGNAL(triggered()), this, SLOT(computeAttributHorizon()));
	}
	if(m_name == "Nurbs" )
	{
		//QAction *addNurbsAction = new QAction("Add Nurbs", this);
		QAction *informationAction = new QAction("Information", this);
	//	QAction *removeNurbsAction = new QAction("Remove Nurbs", this);

	//	menu->addAction(addNurbsAction);
		menu->addAction(informationAction);
	//	menu->addAction(removeNurbsAction);

	//	connect(addNurbsAction, SIGNAL(triggered()), this, SLOT(addData()));
		connect(informationAction, SIGNAL(triggered()), this, SLOT(openNurbsInformation()));
	//	connect(removeNurbsAction, SIGNAL(triggered()), this, SLOT(removeNurbs()));
	}
	if(m_name == "Horizons Animation" )
		{
			//QAction *addNurbsAction = new QAction("Add Nurbs", this);
			QAction *informationAction = new QAction("Information", this);
		//	QAction *removeNurbsAction = new QAction("Remove Nurbs", this);

		//	menu->addAction(addNurbsAction);
			menu->addAction(informationAction);
		//	menu->addAction(removeNurbsAction);

		//	connect(addNurbsAction, SIGNAL(triggered()), this, SLOT(addData()));
			connect(informationAction, SIGNAL(triggered()), this, SLOT(openInformationHorizons()));
		//	connect(removeNurbsAction, SIGNAL(triggered()), this, SLOT(removeNurbs()));
		}
}

void FolderDataRep::addData(){
	qDebug()<<" ADD DATA : "<<m_name;
	DataUpdatorDialog *dialog = new DataUpdatorDialog(m_name, m_data->workingSetManager(), nullptr);
	dialog->show();
}

void FolderDataRep::computeReflectivity() {
	ComputeReflectivityWidget* widget = new ComputeReflectivityWidget(m_data);
	widget->show();
}

AbstractGraphicRep::TypeRep FolderDataRep::getTypeGraphicRep() {
	return AbstractGraphicRep::NotDefined;
}


void FolderDataRep::addSismageHorizon()
{
	ImportSismageHorizonDialog *p = new ImportSismageHorizonDialog(m_data->workingSetManager());
	p->show();
}


void FolderDataRep::computeAttributHorizon()
{
	HorizonAttributComputeDialog *p = new HorizonAttributComputeDialog(nullptr);
	p->setProjectManager(m_data->workingSetManager()->getManagerWidget());
	p->setWorkingSetManager(m_data->workingSetManager());
	p->show();
}

void FolderDataRep::playAnimation()
{
	HorizonFolderData* folderdata= new HorizonFolderData(m_data->workingSetManager(),"Horizon animation");
	m_data->workingSetManager()->addFolderData(folderdata);

}

void FolderDataRep::openNVHorizonInformation()
{
	NextvisionHorizonInformationAggregator* aggregator = new NextvisionHorizonInformationAggregator(m_data->workingSetManager());
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}


void FolderDataRep::openIsoHorizonInformation()
{
	IsoHorizonInformationAggregator* aggregator = new IsoHorizonInformationAggregator(m_data->workingSetManager());
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}

void FolderDataRep::loadAnimation()
{
	// QString path=m_data->workingSetManager()->getManagerWidget()->get_survey_fullpath_name()+"ImportExport/IJK/HORIZONS/" + QString::fromStdString(FreeHorizonManager::BaseDirectory) + "/Animations/";
	QString path=m_data->workingSetManager()->getManagerWidget()->get_survey_fullpath_name()+"ImportExport/IJK/HORIZONS/" + QString::fromStdString(FreeHorizonManager::BaseDirectory) + "/Animations/";
	//qDebug()<<" PATH =="<<path;

	widgetAnimation* widget = new widgetAnimation(path);
	if ( widget->exec() == QDialog::Accepted)
	{
		int index = widget->m_listeAnim->currentRow();
		if(index>= 0)
		{
			QString nameSelected = widget->m_listeAnim->item(index)->text();

			readAnimation(path+nameSelected+".hor");

			HorizonFolderData* folderdata= new HorizonFolderData(m_data->workingSetManager(),nameSelected);
			m_data->workingSetManager()->addFolderData(folderdata);
		}

	}


}

void FolderDataRep::readAnimation(QString path)
{
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"Read animation ouverture du fichier impossible :"<<path;
		return;
	}


	QTextStream in(&file);

	int nbHorizon =0;
	QString attribut;
	bool paletteRGB = true;
	bool lock = true;
	QVector2D rangered,rangegreen,rangeblue,range;
	std::vector<QString> listepath;
	std::vector<QString> listename;
	QList<int > listeindex;
	while(!in.atEnd())
	{
		QString line = in.readLine();
		QStringList linesplit = line.split("|");
		if(linesplit.count()> 0)
		{
			if(linesplit[0] =="nbHorizons")
			{
				nbHorizon= linesplit[1].toInt();

			}
			else if(linesplit[0] =="attributs")
			{
				attribut = linesplit[1];
			}
			else if(linesplit[0] =="typePaletteRGB")
			{
				int RGB = linesplit[1].toInt();
				if(RGB==1 ) paletteRGB = false;
			}
			else if(linesplit[0] =="lockPalette")
			{
				int lockP= linesplit[1].toInt();
				if(lockP==1 ) lock = false;
			}
			else if(linesplit[0] =="rangeRed")
			{
				rangered = QVector2D(linesplit[1].toFloat(),linesplit[2].toFloat());
			}
			else if(linesplit[0] =="rangeGreen")
			{
				rangegreen = QVector2D(linesplit[1].toFloat(),linesplit[2].toFloat());
			}
			else if(linesplit[0] =="rangeBlue")
			{
				rangeblue = QVector2D(linesplit[1].toFloat(),linesplit[2].toFloat());
			}
			else if(linesplit[0] =="range")
			{
				range = QVector2D(linesplit[1].toFloat(),linesplit[2].toFloat());
			}
			else if(linesplit[0] =="path")
			{
				int index = linesplit[1].toInt();
				QString path = linesplit[2];

				QStringList names = path.split("/");
				QString name = names[names.count()-1];
				listepath.push_back(path);
				listename.push_back(name);
				listeindex.push_back(index);
			}
		}
	}
	file.close();

	//qDebug()<<" import animation ok";
	QString surveyPath = m_data->workingSetManager()->getManagerWidget()->get_survey_fullpath_name();
	QString surveyName = m_data->workingSetManager()->getManagerWidget()->get_survey_name();
	qDebug()<<"surveyPath  : "<<surveyPath<<" , " <<surveyName;
	bool bIsNewSurvey = false;
	SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(m_data->workingSetManager(), surveyName, surveyPath, bIsNewSurvey);
	DataSelectorDialog::addNVHorizons(m_data->workingSetManager(), survey, listepath, listename);
}

void FolderDataRep::openPicksInformation()
{
	PickInformationAggregator* aggregator = new PickInformationAggregator(m_data->workingSetManager());
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}

void FolderDataRep::openNurbsInformation()
{
	NurbInformationAggregator* aggregator = new NurbInformationAggregator(m_data->workingSetManager());
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();

}

void FolderDataRep::openInformationHorizons()
{
	HorizonAnimAggregator* aggregator = new HorizonAnimAggregator(m_data->workingSetManager());
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();

}


void FolderDataRep::openWellInformation()
{
	WellInformationAggregator* aggregator = new WellInformationAggregator(m_data->workingSetManager());
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}


#include "horizondatarep.h"
//#include "horizonfolderdata.h"
#include "horizonproppanel.h"
#include "horizonfolder3dlayer.h"
#include "horizonaniminformation.h"

#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "DataUpdatorDialog.h"
#include "DataSelectorDialog.h"
#include "fixedrgblayersfromdatasetandcubeproppanel.h"
#include "fixedrgblayersfromdatasetandcube3dlayer.h"
#include "fixedrgblayersfromdatasetandcubelayeronmap.h"
#include "globalUtil.h"
#include "horizonAttributComputeDialog.h"
#include "horizonfolderlayeronmap.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "importsismagehorizondialog.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "rgbinterleavedqglcudaimageitem.h"
#include "cudargbinterleavedimage.h"
#include "viewqt3d.h"
#include "GeotimeProjectManagerWidget.h"
#include "horizonfolderlayeronslice.h"


//QPointer<HorizonFolderData> HorizonDataRep::m_dataAnim = nullptr;

HorizonDataRep::HorizonDataRep(HorizonFolderData *layer,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_data = layer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_layer3D=nullptr;
	m_name = m_data->name();

	m_image= nullptr;
	m_isoSurfaceHolder = nullptr;


	connect(m_data, SIGNAL(currentChanged()), this,SLOT(dataChanged()));

	connect(m_data,SIGNAL(requestComputeCache()),this,SLOT( computeCache()));
	connect(m_data,SIGNAL(requestClearCache()),this,SLOT( clearCache()));
	connect(m_data,SIGNAL(requestShowCache(int)),this,SLOT( showCache(int)));


}


HorizonDataRep::~HorizonDataRep() {
	if (m_layer3D != nullptr)
		delete m_layer3D;
	if (m_propPanel!=nullptr)
		delete m_propPanel;
	if (m_layer != nullptr)
		delete m_layer;
}


HorizonFolderData* HorizonDataRep::horizonFolderData() const {
	return m_data;
}

void HorizonDataRep::dataChanged() {

	//qDebug()<<"datachanged debut";
	//std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	setBuffer();
//	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//	qDebug() << "datachanged finish ::show: " << std::chrono::duration<double, std::milli>(end-start).count();
}




void HorizonDataRep::buildContextMenu(QMenu *menu) {
	if (m_name == FREE_HORIZON_LABEL ) {
		QAction *addNVHorizonAction = new QAction("Add Nextvision Horizons", this);
		QAction *addSismageHorizonAction = new QAction("Add Sismage Horizons", this);
		QAction *computeFreeHorizonAttibutAction = new QAction("Compute attributs", this);
		menu->addAction(addNVHorizonAction);
		menu->addAction(addSismageHorizonAction);
		menu->addAction(computeFreeHorizonAttibutAction);
		connect(addNVHorizonAction, SIGNAL(triggered()), this, SLOT(addData()));
		connect(addSismageHorizonAction, SIGNAL(triggered()), this, SLOT(addSismageHorizon()));
		connect(computeFreeHorizonAttibutAction, SIGNAL(triggered()), this, SLOT(computeAttributHorizon()));
	}
}

void HorizonDataRep::addData(){
	DataUpdatorDialog *dialog = new DataUpdatorDialog(m_name, m_data->workingSetManager(), nullptr);
	dialog->show();
}


void HorizonDataRep::addSismageHorizon()
{
	ImportSismageHorizonDialog *p = new ImportSismageHorizonDialog(m_data->workingSetManager());
	p->show();
}


void HorizonDataRep::computeAttributHorizon()
{
	HorizonAttributComputeDialog *p = new HorizonAttributComputeDialog(nullptr);
	p->setProjectManager(m_data->workingSetManager()->getManagerWidget());
	p->setWorkingSetManager(m_data->workingSetManager());
	p->show();
}

/*
void HorizonDataRep::dataChanged(FixedRGBLayersFromDatasetAndCube* layer)
{

}*/

IData* HorizonDataRep::data() const {
	return m_data;
}

QWidget* HorizonDataRep::propertyPanel() {

	if (m_propPanel == nullptr)
	{
		m_propPanel = new HorizonPropPanel(this, m_parent);
		if(m_data->currentLayer() != nullptr && (m_parent->viewType()!= ViewType::View3D  || m_data->cubeSeismicAddon().getSampleUnit() == m_sampleUnit))
		{
			if(image() != nullptr) m_propPanel->updatePalette(image());
		}
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* HorizonDataRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new HorizonFolderLayerOnMap(this, scene, defaultZDepth, parent);
	}



	//slice
	//layer : HorizonFolderLayerOnSlice   modele : FixedRGBLayersFromDatasetAndCubeLayerOnSlice
	// rep : HorizonFolderRepOnSlice  modele :FixedRGBLayersFromDatasetAndCubeRepOnSlice
	//panel HorizonFolderPropPanelOnSlice modele :FixedRGBLayersFromDatasetAndCubePropPanelOnSlice

	//random
	//layer : HorizonFolderLayerOnRandom   modele : FixedRGBLayersFromDatasetAndCubeLayerOnRandom
	// rep : HorizonFolderRepOnRandom  modele :FixedRGBLayersFromDatasetAndCubeRepOnRandom
	//panel HorizonFolderPropPanelOnRandom


	//map
	//layer HorizonFolderLayerOnMap	 modele : FixedRGBLayersFromDatasetAndCubeLayerOnMap
	//?? rep : HorizonFolderRepOnMap  modele :FixedRGBLayersFromDatasetAndCubeRepOnMap
	//???panel HorizonFolderPropPanelOnMap

	return m_layer;
}

Graphic3DLayer * HorizonDataRep::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera)
{
	if (m_layer3D == nullptr) {
		m_layer3D = new HorizonFolder3DLayer(this, parent, root, camera);
	}
	return m_layer3D;
}


void HorizonDataRep::clearCache()
{
	if(m_layer3D != nullptr)
	{

		m_layer3D->clearCacheAnimation();
	}
	if(m_image != nullptr)
	{
		m_image->deleteLater();
		m_image=nullptr;
	}

	if(m_isoSurfaceHolder != nullptr)
	{
		m_isoSurfaceHolder->deleteLater();
		m_isoSurfaceHolder= nullptr;
	}

}

void HorizonDataRep::computeCache()
{

	QList<FreeHorizon*> listfree =  horizonFolderData()->completOrderList();

	for(int i=0;i<listfree.count();i++)
	{
		if(m_nameAttribut!="" && listfree[i] != nullptr && ( m_parent->viewType()!= ViewType::View3D  || m_data->cubeSeismicAddon().getSampleUnit() == m_sampleUnit) )
			{



				bool createOKImage =true;// (m_image==nullptr);
				bool createOKIso = true;//(m_isoSurfaceHolder==nullptr);



				int width =  listfree[i]->getLayer(m_nameAttribut).width();
				int height =  listfree[i]->getLayer(m_nameAttribut).depth();
				ImageFormats::QSampleType sample =  listfree[i]->getLayer(m_nameAttribut).imageType();
				ImageFormats::QSampleType sampleIso =  listfree[i]->getLayer(m_nameAttribut).isoType();
			/*	if(!createOKImage)
				{
					if(width != m_image->width() || height != m_image->height() )
					{
						createOKImage = true;
						createOKIso= true;
					}
					if( sample != m_image->sampleType())
					{
						createOKImage= true;
					}
					if( sampleIso != m_isoSurfaceHolder->sampleType())
					{
						createOKIso= true;
					}

				}*/
				CUDARGBInterleavedImage* lastImage = nullptr;
				CPUImagePaletteHolder* lastIsoSurfaceHolder= nullptr;
				if(createOKImage  )
				{
					lastImage = m_image;
					m_image  = new CUDARGBInterleavedImage(width, height,sample , listfree[i]->ijToXYTransfo(m_nameAttribut),this);

				}
				if(createOKIso  )
				{
					lastIsoSurfaceHolder = m_isoSurfaceHolder;
					m_isoSurfaceHolder  = new CPUImagePaletteHolder(width, height,sampleIso , listfree[i]->ijToXYTransfo(m_nameAttribut),this);
				}


				listfree[i]->getLayer(m_nameAttribut).copyImageData(m_image);

		        if(horizonFolderData()->isRangeLocked(m_nameAttribut) == true)
		        {
		        	m_image->setRedRange(horizonFolderData()->lockedRangeRed(m_nameAttribut));
		        	m_image->setGreenRange(horizonFolderData()->lockedRangeGreen(m_nameAttribut));
		        	m_image->setBlueRange(horizonFolderData()->lockedRangeBlue(m_nameAttribut));
		        }

		        listfree[i]->getLayer(m_nameAttribut).copyIsoData(m_isoSurfaceHolder);

				if(lastImage != nullptr) lastImage->deleteLater();
				if(lastIsoSurfaceHolder != nullptr) lastIsoSurfaceHolder->deleteLater();
				//	m_currentLayer->isoSurfaceHolder()->unlockPointer();


				if (m_propPanel != nullptr)
				{
					m_propPanel->updatePalette(image());


				}
				/*if(m_layer)
				{
					m_layer->setBuffer(image(),isoSurfaceHolder());
				}*/

				if(m_layer3D != nullptr)
				{

					m_layer3D->generateCacheAnimation(m_image,m_isoSurfaceHolder);
					QCoreApplication::processEvents();

					/*if(m_image != nullptr)
					{
						m_image->deleteLater();
						m_image=nullptr;
					}

					if(m_isoSurfaceHolder != nullptr)
					{
						m_isoSurfaceHolder->deleteLater();
						m_isoSurfaceHolder= nullptr;
					}*/
				}
			}
			else
			{
				if(m_layer3D != nullptr) m_layer3D->setBuffer(nullptr,nullptr);
				if(m_layer!= nullptr) m_layer->setBuffer(nullptr,nullptr);
				if (m_propPanel != nullptr) m_propPanel->updatePalette(nullptr);
			}
	}
}

void HorizonDataRep::showCache(int i)
{
	if(m_layer3D != nullptr)
	{

		m_layer3D->setVisible(i);
	}
}


HorizonDataRep::HorizonAnimParams HorizonDataRep::readAnimationHorizon(QString path,bool* ok)
{
	HorizonDataRep::HorizonAnimParams params;
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"Read animation ouverture du fichier impossible :"<<path;

		// reset params if read failed
		params.attribut = "";
		params.nbHorizons = 0;
		params.lockPalette = false;
		params.paletteRGB = true;
		params.rangeR = QVector2D(0, 0);
		params.rangeG = QVector2D(0, 0);
		params.rangeB = QVector2D(0, 0);
		params.horizons.clear();
		params.orderIndex.clear();

		if (ok!=nullptr)
		{
			*ok = false;
		}
		return params;;
	}


	QTextStream in(&file);

	bool valid = true;
	int nbHorizon =0;
	QString attribut;
	bool paletteRGB = true;
	bool lock = true;
	QVector2D rangered,rangegreen,rangeblue,range;
	QList<QString > listepath;
	QList<int > listeindex;
	while(!in.atEnd() && valid)
	{
		QString line = in.readLine();
		QStringList linesplit = line.split("|");
		if(linesplit.count()> 0)
		{
			if(linesplit[0] =="nbHorizons" && linesplit.count()> 1)
			{
				nbHorizon= linesplit[1].toInt(&valid);
				params.nbHorizons = nbHorizon;

			}
			else if(linesplit[0] =="attributs" && linesplit.count()> 1)
			{
				attribut = linesplit[1];
				params.attribut =  linesplit[1];
			}
			else if(linesplit[0] =="typePaletteRGB" && linesplit.count()> 1)
			{
				int RGB = linesplit[1].toInt(&valid);
				if(RGB==0 ) paletteRGB = false;
				params.paletteRGB = paletteRGB;
			}
			else if(linesplit[0] =="lockPalette" && linesplit.count()> 1)
			{
				int lockP= linesplit[1].toInt(&valid);
				if(lockP==0 ) lock = false;
				params.lockPalette = lock;
			}
			else if(linesplit[0] =="rangeRed" && linesplit.count()> 2)
			{
				bool test1, test2;
				rangered = QVector2D(linesplit[1].toFloat(&test1),linesplit[2].toFloat(&test2));
				params.rangeR = rangered;
				valid = test1 && test2;
			}
			else if(linesplit[0] =="rangeGreen" && linesplit.count()> 2)
			{
				bool test1, test2;
				rangegreen = QVector2D(linesplit[1].toFloat(&test1),linesplit[2].toFloat(&test2));
				params.rangeG = rangegreen;
				valid = test1 && test2;
			}
			else if(linesplit[0] =="rangeBlue" && linesplit.count()> 2)
			{
				bool test1, test2;
				rangeblue = QVector2D(linesplit[1].toFloat(&test1),linesplit[2].toFloat(&test2));
				params.rangeB = rangeblue;
				valid = test1 && test2;
			}
			else if(linesplit[0] =="range" && linesplit.count()> 2)
			{
				bool test1, test2;
				range = QVector2D(linesplit[1].toFloat(&test1),linesplit[2].toFloat(&test2));
				params.rangeR = range;
				valid = test1 && test2;
			}
			else if(linesplit[0] =="path" && linesplit.count()> 2)
			{
				int index = linesplit[1].toInt(&valid);
				QString path = linesplit[2];
				params.horizons.push_back(linesplit[2]);
				params.orderIndex.push_back(index);

				listepath.push_back(path);
				listeindex.push_back(index);
			}
		}
	}
	file.close();

	// check nbHorizons
	valid = valid && params.nbHorizons == params.horizons.size() && params.nbHorizons == params.orderIndex.size();
	if (valid)
	{
		// check orderIndex
		std::vector<bool> usedIndexes;
		usedIndexes.resize(params.orderIndex.size(), false);

		int idx = 0;
		while (valid && idx<params.orderIndex.size())
		{
			int orderIdx = params.orderIndex[idx];
			valid = orderIdx>=0 && orderIdx<usedIndexes.size() && usedIndexes[orderIdx]==false;
			usedIndexes[orderIdx] = true;
			idx++;
		}
	}

	if (!valid)
	{
		// reset params if read failed
		params.attribut = "";
		params.nbHorizons = 0;
		params.lockPalette = false;
		params.paletteRGB = true;
		params.rangeR = QVector2D(0, 0);
		params.rangeG = QVector2D(0, 0);
		params.rangeB = QVector2D(0, 0);
		params.horizons.clear();
		params.orderIndex.clear();
	}

	if (ok!=nullptr)
	{
		*ok = valid;
	}
	return params;
}

/*
QStringList HorizonDataRep::getAttributesAvailable(HorizonFolderData* data)
{
	QList<FreeHorizon*> listFree =m_data->completOrderList();
	QStringList listAttributs;


	for(int i=1;i<listFree.count();i++)
	{
		for(int j=0;j< listFree[i]->m_attribut.size();j++)
		{
			QString name = listFree[i]->m_attribut[j].name();
			if(name!="" && ! listAttributs.contains(name))
			{

				listAttributs.append(name);
			}
		}
	}
	return listAttributs;
}*/

bool HorizonDataRep::writeAnimationHorizon(QString path, HorizonAnimParams params)
{
	QFile file(path);
	if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		qDebug()<<" ouverture du fichier impossible "<<path;
		return false;
	}
	QTextStream out(&file);

	/*int nbHorizon = m_rep->horizonFolderData()->getNbFreeHorizon();
	if(m_rep->getNameAttribut()=="")
	{
		qDebug()<<"Error attribut not selected";
		return;
	}*/
	//qDebug()<<"m_rep->getNameAttribut()==>"<<params.nameAttribut;
	out<<"nbHorizons"<<"|"<<params.nbHorizons<<"\n";
	out<<"lockPalette"<<"|"<<params.lockPalette<<"\n";
	out<<"attributs"<<"|"<<params.attribut<<"\n";

	out<<"typePaletteRGB"<<"|"<<params.paletteRGB<<"\n";
	if(params.paletteRGB)
	{
		QVector2D range1 =params.rangeR;// m_paletteRGB->getRange(0);//  m_rep->horizonFolderData()->lockedRangeRed(m_rep->getNameAttribut());
		out<<"rangeRed"<<"|"<<range1.x()<<"|"<<range1.y()<<"\n";
		QVector2D range2 = params.rangeG;// m_paletteRGB->getRange(1);//m_rep->horizonFolderData()->lockedRangeGreen(m_rep->getNameAttribut());
		out<<"rangeGreen"<<"|"<<range2.x()<<"|"<<range2.y()<<"\n";
		QVector2D range3 = params.rangeB;// m_paletteRGB->getRange(2);//m_rep->horizonFolderData()->lockedRangeBlue(m_rep->getNameAttribut());
		out<<"rangeBlue"<<"|"<<range3.x()<<"|"<<range3.y()<<"\n";
	}
	else
	{
		QVector2D range1 = params.rangeR;// m_palette->getRange();
		out<<"range"<<"|"<<range1.x()<<"|"<<range1.y()<<"\n";
	}
	for(int i=0;i<params.nbHorizons;i++)
	{
		out<<"path"<<"|"<<params.orderIndex[i]<<"|"<<params.horizons[i]<<"\n";
	}

	out<<"\n";

	file.close();

	return true;
}

void HorizonDataRep::addAnimationHorizon(QString path,QString name, WorkingSetManager* manager)
{


	//std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	bool valid;
	HorizonDataRep::HorizonAnimParams params = readAnimationHorizon(path,&valid);


	if(valid)
	{
		std::vector<QString> listepath;
		std::vector<QString> listename;

		for(int i=0;i<params.horizons.size();i++)
		{
			QFileInfo fileinfo(params.horizons[i]);
			listepath.push_back(params.horizons[i]);
			listename.push_back(fileinfo.baseName());

		}
		QString surveyPath =manager->getManagerWidget()->get_survey_fullpath_name();
		QString surveyName = manager->getManagerWidget()->get_survey_name();

		bool bIsNewSurvey = false;
		SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(manager, surveyName, surveyPath, bIsNewSurvey);
		DataSelectorDialog::addNVHorizons(manager, survey, listepath, listename);

	//	m_dataAnim = new HorizonFolderData(manager,name);
	//	manager->addHorizonAnimData(m_dataAnim);
	}

	//	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//	qDebug() << "HorizonDataRep finish ::show: " << std::chrono::duration<double, std::milli>(end-start).count();
}

void HorizonDataRep::removeAnimationHorizon(QString path ,QString name,WorkingSetManager* manager)
{
	//manager->removeHorizonAnimData(horizonfolderdata);
}

HorizonAnimInformation* HorizonDataRep::newAnimationHorizon(WorkingSetManager* manager )
{
	widgetNameForSave* widget = new widgetNameForSave("Horizon animation",nullptr);
	if ( widget->exec() == QDialog::Accepted)
	{
		QString nom = widget->getName();

		if(nom!="" )
		{
			DataUpdatorDialog *dialog = new DataUpdatorDialog("Horizon", manager, nullptr);
			dialog->setForceAllItems(true);
			if ( dialog->exec() == QDialog::Accepted)
			{

				std::vector<QString> horizons = dialog->getPathSelected();
				//m_dataAnim = new HorizonFolderData(manager,nom);
				//manager->addHorizonAnimData(m_dataAnim);
				QString dataFullname = manager->getManagerWidget()->get_horizonanim_path0();
				dataFullname += nom + ".hor";

				HorizonAnimInformation* info = new HorizonAnimInformation(nom, dataFullname,horizons, manager);
				return info;
			}
		}
	}

	return nullptr;
}



void HorizonDataRep::setBuffer()
{



	if(m_nameAttribut!="" && m_data->currentLayer() != nullptr && ( m_parent->viewType()!= ViewType::View3D  || m_data->cubeSeismicAddon().getSampleUnit() == m_sampleUnit) )
	{

		//CUDARGBInterleavedImage* imageCurrent  = m_data->currentLayer()->image(m_nameAttribut);
		//CPUImagePaletteHolder* isoCurrent =m_data->currentLayer()->isoSurface(m_nameAttribut);

	//	if(m_data->currentLayer()==nullptr) return;

		bool createOKImage = (m_image==nullptr);
		bool createOKIso = (m_isoSurfaceHolder==nullptr);



		int width = m_data->currentLayer()->getLayer(m_nameAttribut).width();
		int height = m_data->currentLayer()->getLayer(m_nameAttribut).depth();
		ImageFormats::QSampleType sample = m_data->currentLayer()->getLayer(m_nameAttribut).imageType();
		ImageFormats::QSampleType sampleIso = m_data->currentLayer()->getLayer(m_nameAttribut).isoType();
		if(!createOKImage)
		{
			if(width != m_image->width() || height != m_image->height() )
			{
				createOKImage = true;
				createOKIso= true;
			}
			if( sample != m_image->sampleType())
			{
				createOKImage= true;
			}
			if( sampleIso != m_isoSurfaceHolder->sampleType())
			{
				createOKIso= true;
			}

		}
		CUDARGBInterleavedImage* lastImage = nullptr;
		CPUImagePaletteHolder* lastIsoSurfaceHolder= nullptr;
		if(createOKImage  )
		{
			lastImage = m_image;
			m_image  = new CUDARGBInterleavedImage(width, height,sample ,m_data->currentLayer()->ijToXYTransfo(m_nameAttribut),this);

		}
		if(createOKIso  )
		{
			lastIsoSurfaceHolder = m_isoSurfaceHolder;
			m_isoSurfaceHolder  = new CPUImagePaletteHolder(width, height,sampleIso ,m_data->currentLayer()->ijToXYTransfo(m_nameAttribut),this);
		}


        m_data->currentLayer()->getLayer(m_nameAttribut).copyImageData(m_image);

        if(horizonFolderData()->isRangeLocked(m_nameAttribut) == true)
        {
        	m_image->setRedRange(horizonFolderData()->lockedRangeRed(m_nameAttribut));
        	m_image->setGreenRange(horizonFolderData()->lockedRangeGreen(m_nameAttribut));
        	m_image->setBlueRange(horizonFolderData()->lockedRangeBlue(m_nameAttribut));
        }

        m_data->currentLayer()->getLayer(m_nameAttribut).copyIsoData(m_isoSurfaceHolder);

		if(lastImage != nullptr) lastImage->deleteLater();
		if(lastIsoSurfaceHolder != nullptr) lastIsoSurfaceHolder->deleteLater();
		//	m_currentLayer->isoSurfaceHolder()->unlockPointer();


		if (m_propPanel != nullptr)
		{
			m_propPanel->updatePalette(image());


		}
		if(m_layer)
		{
			m_layer->setBuffer(image(),isoSurfaceHolder());
		}

		if(m_layer3D != nullptr)
		{

			m_layer3D->setBuffer(image(),isoSurfaceHolder());
		}
	}
	else
	{
		if(m_layer3D != nullptr) m_layer3D->setBuffer(nullptr,nullptr);
		if(m_layer!= nullptr) m_layer->setBuffer(nullptr,nullptr);
		if (m_propPanel != nullptr) m_propPanel->updatePalette(nullptr);
	}

}


bool HorizonDataRep::mouseData(double x, double y, MouseInfo &info) {
	double v1=0, v2=0, v3=0;

	if(image()== nullptr || isoSurfaceHolder() == nullptr || m_data->sampleTransformation() == nullptr)
	{
		return false;
	}

	bool valid = image()->value(x, y, 0,
			info.i, info.j, v1);
	valid = valid && image()->value(x, y, 1,
			info.i, info.j, v2);
	valid = valid && image()->value(x, y, 2,
			info.i, info.j, v3);

	info.values.push_back(v1);
	info.values.push_back(v2);
	info.values.push_back(v3);

	info.valuesDesc.push_back("Red");
	info.valuesDesc.push_back("Green");
	info.valuesDesc.push_back("Blue");
	info.depthValue = true;

	IGeorefImage::value(isoSurfaceHolder(), x, y, info.i, info.j,
			v1);
	double realDepth;
	m_data->sampleTransformation()->direct(v1, realDepth);
	info.depth = realDepth;
	info.depthUnit = m_data->cubeSeismicAddon().getSampleUnit();

	return valid;
}

bool HorizonDataRep::setSampleUnit(SampleUnit sampleUnit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	m_sampleUnit = sampleUnit;
	return list.contains(sampleUnit);
}

QList<SampleUnit> HorizonDataRep::getAvailableSampleUnits() const {

	CubeSeismicAddon addon = m_data->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());

	return list;
}

QString HorizonDataRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep HorizonDataRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}

void HorizonDataRep::deleteGraphicItemDataContent(QGraphicsItem *item)
{
	deleteData(image(),item);
	deleteData(isoSurfaceHolder(),item);
}

QGraphicsObject* HorizonDataRep::cloneCUDAImageWithMask(QGraphicsItem *parent)
{
	RGBInterleavedQGLCUDAImageItem* outItem = new RGBInterleavedQGLCUDAImageItem(isoSurfaceHolder(),
			image(), 0,
				parent, true);

//	outItem->setMinimumValue(m_data->minimumValue());
//	outItem->setMinimumValueActive(m_data->isMinimumValueActive());

	return outItem;
}


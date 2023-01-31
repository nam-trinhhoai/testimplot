#include "workingsetmanager.h"

#include "GraphicTool_GraphicLayer.h"
#include "idata.h"
#include "seismicsurvey.h"
#include "stratislice.h"
#include "LayerSlice.h"
#include "rgblayerslice.h"
#include "multiseedhorizon.h"
// #include "multiseedrgt.h"
#include "rgbstratisliceattribute.h"
#include "fixedlayerfromdataset.h"
#include "fixedrgblayersfromdataset.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "fixedlayersfromdatasetandcube.h"
#include "ijkhorizon.h"
#include "wellhead.h"
#include "marker.h"
#include "freehorizon.h"
#include "isohorizon.h"
#include "folderdata.h"
#include "horizonfolderdata.h"
#include "videolayer.h"
#include "rgbdataset.h"
#include "rgblayerfromdataset.h"
#include "rgbcomputationondataset.h"
#include "nurbsdataset.h"
#include "randomdataset.h"
#include "randomtexdataset.h"
#include "scalarComputationOnDataSet.h"
#include "computationoperatordataset.h"
#include "nurbswidget.h"

#include <globalUtil.h>


WorkingSetManager::WorkingSetManager(QObject *parent):QObject(parent)
{
	m_folders.seismics = new FolderData(this, "Seismics", this);
	m_folders.wells = new FolderData(this, "Wells", this);
	m_folders.markers = new FolderData(this, "Markers", this);
	m_folders.graphicLayers = new FolderData(this, "Graphic Layers", this);
	m_folders.nurbs = new FolderData(this, "Nurbs", this);
	m_folders.randoms = new FolderData(this, "Randoms", this);
	m_folders.horizonsIso = new FolderData(this, NV_ISO_HORIZON_LABEL, this);
	m_folders.horizonsFree = new FolderData(this, FREE_HORIZON_LABEL, this);
	m_folders.horizonsAnim = new FolderData(this, "Horizons Animation", this);
	m_folders.others = new FolderData(this, "Others", this);
	addData(m_folders.seismics);
	addData(m_folders.wells);
	addData(m_folders.markers);
	addData(m_folders.graphicLayers);
	addData(m_folders.nurbs);
	addData(m_folders.randoms);
	addData(m_folders.horizonsIso);
	addData(m_folders.horizonsFree);
	addData(m_folders.horizonsAnim);
	addData(m_folders.others);
}
WorkingSetManager::~WorkingSetManager()
{
	//m_folders FolderData delete taken care of by QObject default mechanism
}

void WorkingSetManager::addSeismicSurvey(SeismicSurvey *survey)
{
	m_folders.seismics->addData(survey);
}

void WorkingSetManager::addStratiSlice(StratiSlice *stratiSlice)
{
	m_folders.horizonsFree->addData(stratiSlice);
}

void WorkingSetManager::addLayerSlice(LayerSlice *layerSlice)
{
	m_folders.horizonsFree->addData(layerSlice);
}

void WorkingSetManager::addRGBLayerSlice(RGBLayerSlice *layerSlice)
{
	m_folders.horizonsFree->addData(layerSlice);
}

void WorkingSetManager::addRGBStratiSlice(RGBStratiSliceAttribute *stratiSlice)
{
	m_folders.horizonsFree->addData(stratiSlice);
}

void WorkingSetManager::addMultiSeedHorizon(MultiSeedHorizon *horizon)
{
	m_folders.horizonsFree->addData(horizon);
}

void WorkingSetManager::addFixedLayerFromDataset(FixedLayerFromDataset* layer) {
	m_folders.horizonsFree->addData(layer);
}

void WorkingSetManager::addFixedRGBLayersFromDataset(FixedRGBLayersFromDataset* layer) {
	m_folders.horizonsFree->addData(layer);
}

void WorkingSetManager::addFixedRGBLayersFromDatasetAndCube(FixedRGBLayersFromDatasetAndCube* layer) {
	m_folders.horizonsIso->addData(layer);
}

void WorkingSetManager::addFixedLayersFromDatasetAndCube(FixedLayersFromDatasetAndCube* layer) {
	m_folders.horizonsIso->addData(layer);
}

void WorkingSetManager::addIJKHorizon(IJKHorizon* horizon) {
	m_folders.horizonsFree->addData(horizon);
}

void WorkingSetManager::addWellHead(WellHead* wellHead) {
	m_folders.wells->addData(wellHead);
}

void WorkingSetManager::addMarker(Marker* marker) {
	m_folders.markers->addData(marker);
}

void WorkingSetManager::addVideoLayer(VideoLayer* layer) {
	m_folders.horizonsIso->addData(layer);
}

void WorkingSetManager::addRgbDataset(RgbDataset* rgbDataset) {
	m_folders.seismics->addData(rgbDataset);
}

void WorkingSetManager::addRgbLayerFromDataset(RgbLayerFromDataset* layer) {
	m_folders.horizonsFree->addData(layer);
}

void WorkingSetManager::addRgbComputationOnDataset(RgbComputationOnDataset* dataset) {
	m_folders.seismics->addData(dataset);
}

void WorkingSetManager::addScalarComputationOnDataset(ScalarComputationOnDataset* dataset) {
	m_folders.seismics->addData(dataset);
}

void WorkingSetManager::addComputationOperatorDataset(ComputationOperatorDataset* dataset) {
	m_folders.seismics->addData(dataset);
}


void WorkingSetManager::addGraphicLayer(GraphicTool_GraphicLayer* dataset) {
	m_folders.graphicLayers->addData(dataset);
}


void WorkingSetManager::addNurbs(NurbsDataset* dataset) {
	m_folders.nurbs->addData(dataset);
}

void WorkingSetManager::addRandom(RandomDataset* dataset) {
	m_folders.randoms->addData(dataset);
}


void WorkingSetManager::addFreeHorizons(FreeHorizon* layer){
	m_folders.horizonsFree->addData(layer);
}

void WorkingSetManager::addIsoHorizons(IsoHorizon* layer){
	m_folders.horizonsIso->addData(layer);
}

void WorkingSetManager::removeFreeHorizons(FreeHorizon* layer){
	m_folders.horizonsFree->removeData(layer);
}

void WorkingSetManager::removeIsoHorizons(IsoHorizon* layer){
	m_folders.horizonsIso->removeData(layer);
}


void WorkingSetManager::addHorizonsIsoFromDirectories(FixedRGBLayersFromDatasetAndCube* layer) {
	m_folders.horizonsIso->addData(layer);
}

void WorkingSetManager::addHorizonsFreeFromDirectories(FixedRGBLayersFromDatasetAndCube* layer) {
	m_folders.horizonsFree->addData(layer);
}


void WorkingSetManager::addFolderData(HorizonFolderData * horizonfolderdata)
{
	m_folders.horizonsFree->addData(horizonfolderdata);
}


void WorkingSetManager::removeFolderData(HorizonFolderData * horizonfolderdata)
{
	m_folders.horizonsFree->removeData(horizonfolderdata);
}

void WorkingSetManager::addHorizonAnimData(HorizonFolderData * horizonfolderdata)
{
	m_folders.horizonsAnim->addData(horizonfolderdata);
}
void WorkingSetManager::removeHorizonAnimData(HorizonFolderData * horizonfolderdata)
{
	m_folders.horizonsAnim->removeData(horizonfolderdata);
}


bool WorkingSetManager::containsFreeHorizon(QString path)
{
	FolderData* folder = m_folders.horizonsFree;
	QList<IData*> datas = folder->data();
	for(int i=0;i<datas.size();i++)
	{
		FreeHorizon* free = dynamic_cast<FreeHorizon*>(datas[i]);
		if(free != nullptr && free->path() == path) return true;
	}
	return false;
}

FreeHorizon *WorkingSetManager::getFreeHorizon(QString path)
{
	FolderData* folder = m_folders.horizonsFree;
	QList<IData*> datas = folder->data();
	for(int i=0;i<datas.size();i++)
	{
		FreeHorizon* free = dynamic_cast<FreeHorizon*>(datas[i]);
		if(free != nullptr && free->path() == path) return free;
	}
	return nullptr;
}

bool WorkingSetManager::containsHorizonAnim(const QString& name)
{
	FolderData* folder = m_folders.horizonsAnim;
	QList<IData*> datas = folder->data();
	for(int i=0;i<datas.size();i++)
	{
		HorizonFolderData* anim = dynamic_cast<HorizonFolderData*>(datas[i]);
		if(anim != nullptr && anim->name() == name) return true;
	}
	return false;
}

/*
void WorkingSetManager::addMultiSeedRgt(MultiSeedRgt *rgt)
{
	m_folders.horizonsFree->addData(rgt);
}
*/

void WorkingSetManager::removeSeismicSurvey(SeismicSurvey *survey)
{
	m_folders.seismics->removeData(survey);
}

void WorkingSetManager::removeStratiSlice(StratiSlice *stratiSlice)
{
	m_folders.horizonsFree->removeData(stratiSlice);
}

void WorkingSetManager::removeLayerSlice(LayerSlice *layerSlice)
{
	m_folders.horizonsFree->removeData(layerSlice);
}

// 17082021
void WorkingSetManager::deleteLayerSlice(LayerSlice *layerSlice)
{
	m_folders.horizonsFree->deleteData(layerSlice);
}

void WorkingSetManager::removeRGBLayerSlice(RGBLayerSlice *layerSlice)
{
	m_folders.horizonsFree->removeData(layerSlice);
}
// 17082021
void WorkingSetManager::deleteRGBLayerSlice(RGBLayerSlice *layerSlice)
{
	m_folders.horizonsFree->deleteData(layerSlice);
}

void WorkingSetManager::removeRGBStratiSlice(RGBStratiSliceAttribute *stratiSlice)
{
	m_folders.horizonsFree->removeData(stratiSlice);
}

void WorkingSetManager::removeMultiSeedHorizon(MultiSeedHorizon *horizon)
{
	m_folders.horizonsFree->removeData(horizon);
}

void WorkingSetManager::removeFixedLayerFromDataset(FixedLayerFromDataset* layer) {
	m_folders.horizonsFree->removeData(layer);
}

void WorkingSetManager::removeFixedRGBLayersFromDataset(FixedRGBLayersFromDataset* layer) {
	m_folders.horizonsFree->removeData(layer);
}

void WorkingSetManager::removeFixedRGBLayersFromDatasetAndCube(FixedRGBLayersFromDatasetAndCube* layer) {
	m_folders.horizonsIso->removeData(layer);
}

void WorkingSetManager::removeFixedLayersFromDatasetAndCube(FixedLayersFromDatasetAndCube* layer) {
	m_folders.horizonsIso->removeData(layer);
}

void WorkingSetManager::removeIJKHorizon(IJKHorizon* horizon) {
	m_folders.horizonsFree->removeData(horizon);
}

void WorkingSetManager::removeWellHead(WellHead* wellHead) {
	m_folders.wells->removeData(wellHead);
}

// 17082021
void WorkingSetManager::deleteWellHead(WellHead* wellHead) {
	m_folders.wells->deleteData(wellHead);
}

void WorkingSetManager::removeMarker(Marker* marker) {
	m_folders.markers->removeData(marker);
}

void WorkingSetManager::deleteMarker(Marker* marker) {
	m_folders.markers->deleteData(marker);
}

void WorkingSetManager::removeVideoLayer(VideoLayer* layer) {
	m_folders.horizonsIso->removeData(layer);
}

void WorkingSetManager::removeRgbDataset(RgbDataset* rgbDataset) {
	m_folders.seismics->removeData(rgbDataset);
}

void WorkingSetManager::removeRgbLayerFromDataset(RgbLayerFromDataset* layer) {
	m_folders.horizonsFree->removeData(layer);
}

void WorkingSetManager::removeRgbComputationOnDataset(RgbComputationOnDataset* dataset) {
	m_folders.seismics->removeData(dataset);
}

void WorkingSetManager::removeComputationOperatorDataset(ComputationOperatorDataset* dataset) {
	m_folders.seismics->removeData(dataset);
}

void WorkingSetManager::removeGraphicLayer(GraphicTool_GraphicLayer* dataset) {
	m_folders.graphicLayers->addData(dataset);
}

void WorkingSetManager::removeNurbs(NurbsDataset* dataset) {
	m_folders.nurbs->removeData(dataset);
}

void WorkingSetManager::removeRandom(RandomDataset* dataset) {
	m_folders.randoms->removeData(dataset);
}

bool WorkingSetManager::containsLayerSlice(LayerSlice *layer) {
	return m_folders.horizonsFree->isDataContains(layer);
}

/*
void WorkingSetManager::removeMultiSeedRgt(MultiSeedRgt *rgt)
{
	m_folders.horizonsFree->removeData(rgt);
}
*/

void WorkingSetManager::addData(IData *data)
{
	m_data.push_back(data);
	emit dataAdded(data);
}
void WorkingSetManager::removeData(IData *data)
{
	m_data.removeOne(data);
	emit dataRemoved(data);
	delete data;
}

QList<IJKHorizon*> WorkingSetManager::listIJKHorizons() {
	QList<IJKHorizon*> out;
	QList<IData*> layerData = m_folders.horizonsFree->data();
	for (std::size_t i=0; i<layerData.count(); i++) {
		IJKHorizon* horizon = dynamic_cast<IJKHorizon*>(layerData[i]);
		if (horizon!=nullptr) {
			out.push_back(horizon);
		}
	}
	return out;
}

QList<WellHead*> WorkingSetManager::listWellHead() {
	QList<WellHead*> out;
	QList<IData*> wellHeadData = m_folders.wells->data();
	for (std::size_t i=0; i<wellHeadData.count(); i++) {
		WellHead* wellHead = dynamic_cast<WellHead*>(wellHeadData[i]);
		if (wellHead!=nullptr) {
			out.push_back(wellHead);
		}
	}
	return out;
}

void WorkingSetManager::setManagerWidget(GeotimeProjectManagerWidget* pDialog){
	if(pDialog != nullptr){
		m_DiaglogSelector = pDialog;
	}
}


QList<QString> WorkingSetManager::getDataset(RandomDataset* dataset)
{
	QList<QString> listename;
	for(int i=0;i< m_folders.seismics->data().size();i++)
	{
			SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(m_folders.seismics->data().at(i));
			if(survey != nullptr)
			{
				QList<Seismic3DAbstractDataset*> datasets = survey->datasets();

				for(int j = 0;j<datasets.size();j++)
				{

					listename.append(datasets[j]->name());
					/*RandomTexDataset* randomtex = new RandomTexDataset(this,datasets[j]->name(),this);
					dataset->addDataset(randomtex);*/
				}

			}
	}

	return listename;

}


void WorkingSetManager::addNurbs(QString path,QString name)
{
	NurbsWidget::addNurbs(path,name);
}

//void WorkingSetManager::set_horizons(std::vector<QString> horizonNames, std::vector<QString> horizonPaths,
//		std::vector<QString> horizonExtractionDataPaths)
//{
//	int N = horizonNames.size();
//	m_horizonNames.resize(N);
//	m_horizonPaths.resize(N);
//	m_horizonExtractionDataPaths.resize(N);
//	for (int i=0; i<N; i++)
//	{
//		m_horizonNames[i] = horizonNames[i];
//		m_horizonPaths[i] = horizonPaths[i];
//		m_horizonExtractionDataPaths[i] = horizonExtractionDataPaths[i];
//	}
//}

//std::vector<QString> *WorkingSetManager::get_horizonNames()
//{
//	return &m_horizonNames;
//}


//std::vector<QString> *WorkingSetManager::get_horizonPaths()
//{
//	return &m_horizonPaths;
//}


//std::vector<QString> *WorkingSetManager::get_horizonExtractionDataPaths()
//{
//	return &m_horizonExtractionDataPaths;
//}

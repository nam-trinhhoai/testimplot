
/*
#include "wellpickgraphicrepfactory.h"
#include "wellpick.h"
#include "wellpickreponslice.h"
#include "wellpickrep.h"
#include "wellpickreponrandom.h"
*/

#include "seismicsurvey.h"
#include <workingsetmanager.h>
#include <sliceutils.h>

#include "igraphicrepfactory.h"
#include <freehorizonattribut.h>
#include <freehorizonattributreponslice.h>
#include <freehorizonattributlayer.h>
#include <freehorizonattributrepfactory.h>
#include "fixedrgblayersfromdatasetandcube.h"
#include <freeHorizonManager.h>

FreeHorizonAttributRepFactory::FreeHorizonAttributRepFactory(
		FreeHorizonAttribut *data) :
		IGraphicRepFactory(data) {

	m_data = data;
	QString path = m_data->getPath();
	QString name = m_data->getName();
	SeismicSurvey *survey = m_data->getSurvey();
	WorkingSetManager *workingSet = m_data->getWorkingSetManager();

	bool isValid = false;
	QString datasetName = QString::fromStdString(FreeHorizonManager::dataSetNameGet(path.toStdString()));
	QString datasetPath0 = survey->idPath() + "/DATA/SEISMIC/" + datasetName + ".xt";
	FixedRGBLayersFromDatasetAndCube::Grid3DParameter params = FixedRGBLayersFromDatasetAndCube::createGrid3DParameter(datasetPath0, survey, &isValid);

	// connect(m_data, SIGNAL(datasetAdded(Seismic3DAbstractDataset *)), this, SLOT(datasetAdded(Seismic3DAbstractDataset *)));


	// m_layer = new FreeHorizonAttributLayer(path, name, workingSet, survey, params);
}

FreeHorizonAttributRepFactory::~FreeHorizonAttributRepFactory() {

}


AbstractGraphicRep* FreeHorizonAttributRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {

	/*
	if (type == ViewType::InlineView || type == ViewType::XLineView || ViewType::BasemapView || type==ViewType::StackBasemapView || type == ViewType::View3D) {
		return new FreeHorizonAttributRepOnSlice(m_data, parent);
	}
	*/
	if (type == ViewType::InlineView ) {
		return new FreeHorizonAttributRepOnSlice(m_data, SliceDirection::Inline, parent);
	}



	/*else if (type == ViewType::View3D) {
		return new WellPickRep(m_data, parent);
	} else if (type == ViewType::RandomView) {
		return new WellPickRepOnRandom(m_data, parent);
	}
	*/
//	if (type == ViewType::BasemapView || type==ViewType::StackBasemapView)
//		return new WellBoreRepOnMap(m_data, parent);
//	else if (type == ViewType::InlineView || type == ViewType::XLineView) {
//		SliceDirection dir = (type==ViewType::InlineView) ? SliceDirection::Inline : SliceDirection::XLine;
//		return new WellBoreRepOnSlice(m_data, dir, parent);
//	}
//	} else if (type == ViewType::View3D) {
//		return new SeismicSurveyRepOn3D(m_data, parent);
//	}

	return nullptr;
}


void FreeHorizonAttributRepFactory::freeHorizonAttributAdded() {
	// emit childAdded(dataset->graphicRepFactory());
}




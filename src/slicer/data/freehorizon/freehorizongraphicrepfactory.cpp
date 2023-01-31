
#include <freehorizon.h>
#include <freehorizonattribut.h>
#include <freehorizonrep.h>
#include "freehorizongraphicrepfactory.h"
#include <fixedlayerimplfreehorizonfromdataset.h>
#include <fixedlayerimplfreehorizonfromdatasetandcube.h>
#include <rgtSpectrumHeader.h>


FreeHorizonGraphicRepFactory::FreeHorizonGraphicRepFactory(
		FreeHorizon *data) :
		IGraphicRepFactory(data) {
	m_data = data;
	// connect(m_data, SIGNAL(wellPickAdded(WellPick*)), this, SLOT(wellPickAdded(WellPick*)));
	connect(m_data, SIGNAL(attributAdded(FreeHorizon::Attribut *)), this, SLOT(attributAdded(FreeHorizon::Attribut *)));
	connect(m_data, SIGNAL(attributRemoved(FreeHorizon::Attribut *)), this, SLOT(attributRemoved(FreeHorizon::Attribut *)));
}

FreeHorizonGraphicRepFactory::~FreeHorizonGraphicRepFactory() {

}
AbstractGraphicRep* FreeHorizonGraphicRepFactory::rep(ViewType type,AbstractInnerView *parent) {

	return new FreeHorizonRep(m_data, parent);
/*
	if (type == ViewType::InlineView || type == ViewType::XLineView || type==ViewType::View3D || type==ViewType::RandomView) {
		return new FreeHorizonRep(m_data, parent);
	}
	if (type == ViewType::BasemapView || type==ViewType::StackBasemapView)
		return new FreeHorizonRep(m_data, parent);
		*/



	//		return new WellBoreRepOnMap(m_data, parent);
	/*
//	if (type == ViewType::BasemapView || type==ViewType::StackBasemapView)
//		return new WellBoreRepOnMap(m_data, parent);
//	else if (type == ViewType::InlineView || type == ViewType::XLineView) {
//		SliceDirection dir = (type==ViewType::InlineView) ? SliceDirection::Inline : SliceDirection::XLine;
//		return new WellBoreRepOnSlice(m_data, dir, parent);
//	}
//	} else if (type == ViewType::View3D) {
//		return new SeismicSurveyRepOn3D(m_data, parent);
//	}
 * */

	return nullptr;
}

QList<IGraphicRepFactory*> FreeHorizonGraphicRepFactory::childReps(ViewType type,AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;

	for ( FreeHorizon::Attribut d : m_data->m_attribut )
	{
		IGraphicRepFactory * factory = nullptr;
		QString name;
		if ( d.getFixedRGBLayersFromDatasetAndCube() )
		{
			factory = d.getFixedRGBLayersFromDatasetAndCube()->graphicRepFactory();
			name = d.getFixedRGBLayersFromDatasetAndCube()->name();
		}
		else if ( d.getFixedLayerFromDataset() )
		{
			factory = d.getFixedLayerFromDataset()->graphicRepFactory();
			name = d.getFixedLayerFromDataset()->name();
		}
		else if ( d.getFixedLayersImplFreeHorizonFromDatasetAndCube() )
		{
			factory = d.getFixedLayersImplFreeHorizonFromDatasetAndCube()->graphicRepFactory();
			name = d.getFixedLayersImplFreeHorizonFromDatasetAndCube()->name();
		}

		reps.push_back(factory);
	}

	return reps;
}


void FreeHorizonGraphicRepFactory::attributAdded(FreeHorizon::Attribut *data) {
	IGraphicRepFactory * factory = nullptr;
	QString name;

	if ( data->getFixedRGBLayersFromDatasetAndCube() )
	{
		factory = data->getFixedRGBLayersFromDatasetAndCube()->graphicRepFactory();
	}
	else if ( data->getFixedLayerFromDataset() )
	{
		factory = data->getFixedLayerFromDataset()->graphicRepFactory();
		// name = d.getFixedLayerFromDataset()->name();
	}
	else if ( data->getFixedLayersImplFreeHorizonFromDatasetAndCube() )
	{
		factory = data->getFixedLayersImplFreeHorizonFromDatasetAndCube()->graphicRepFactory();
		// name = d.getFixedLayersImplFreeHorizonFromDatasetAndCube()->name();
	}
	if ( factory )
		emit childAdded(factory);
}


void FreeHorizonGraphicRepFactory::attributRemoved(FreeHorizon::Attribut *data)
{
	IGraphicRepFactory * factory = nullptr;
	QString name;

	if ( data->getFixedRGBLayersFromDatasetAndCube() )
	{
		factory = data->getFixedRGBLayersFromDatasetAndCube()->graphicRepFactory();
	}
	else if ( data->getFixedLayerFromDataset() )
	{
		factory = data->getFixedLayerFromDataset()->graphicRepFactory();
		// name = d.getFixedLayerFromDataset()->name();
	}
	else if ( data->getFixedLayersImplFreeHorizonFromDatasetAndCube() )
	{
		factory = data->getFixedLayersImplFreeHorizonFromDatasetAndCube()->graphicRepFactory();
		// name = d.getFixedLayersImplFreeHorizonFromDatasetAndCube()->name();
	}
	if ( factory )
		emit childRemoved(factory);
}

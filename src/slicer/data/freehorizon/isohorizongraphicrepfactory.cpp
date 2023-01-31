
#include <isohorizon.h>
// #include <isohorizonattribut.h>
#include <isohorizonrep.h>
#include "isohorizongraphicrepfactory.h"
#include <rgtSpectrumHeader.h>

IsoHorizonGraphicRepFactory::IsoHorizonGraphicRepFactory(
		IsoHorizon *data) :
		IGraphicRepFactory(data) {
	m_data = data;
	// connect(m_data, SIGNAL(wellPickAdded(WellPick*)), this, SLOT(wellPickAdded(WellPick*)));
}

IsoHorizonGraphicRepFactory::~IsoHorizonGraphicRepFactory() {

}
AbstractGraphicRep* IsoHorizonGraphicRepFactory::rep(ViewType type,AbstractInnerView *parent) {


	if (type == ViewType::InlineView || type == ViewType::XLineView || type==ViewType::View3D || type==ViewType::RandomView) {
		return new IsoHorizonRep(m_data, parent);
	}
	if (type == ViewType::BasemapView || type==ViewType::StackBasemapView)
		return new IsoHorizonRep(m_data, parent);
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

QList<IGraphicRepFactory*> IsoHorizonGraphicRepFactory::childReps(ViewType type,AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;

	/*
	for ( FixedRGBLayersFromDatasetAndCube *d : m_data->m_attribut )
	{
		IGraphicRepFactory * factory = d->graphicRepFactory();
		reps.push_back(factory);
	}
	*/

	for ( IsoHorizon::Attribut d : m_data->m_attribut )
	{
		IGraphicRepFactory * factory = nullptr;
		QString name;
		if ( d.pFixedLayerImplIsoHorizonFromDatasetAndCube )
		{
			factory = d.pFixedLayerImplIsoHorizonFromDatasetAndCube->graphicRepFactory();
			name = d.pFixedLayerImplIsoHorizonFromDatasetAndCube->name();
		}
		else if ( d.pFixedRGBLayersFromDatasetAndCube )
		{
			factory = d.pFixedRGBLayersFromDatasetAndCube->graphicRepFactory();
			name = d.pFixedRGBLayersFromDatasetAndCube->name();
		}


		reps.push_back(factory);
	}

	/*
	for (WellPick *d : m_data->wellPicks()) {
		IGraphicRepFactory * factory= d->graphicRepFactory();
		reps.push_back(factory);
	}
	*/
	return reps;
}

#include "fixedrgblayersfromdatasetandcubegraphicrepfactory.h"

#include "seismic3dabstractdataset.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "cudargbimage.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "affine2dtransformation.h"
#include "seismic3dabstractdataset.h"
#include "abstractinnerview.h"
#include "fixedrgblayersfromdatasetandcuberep.h"
#include "fixedrgblayersfromdatasetandcubereponslice.h"
#include "fixedrgblayersfromdatasetandcubereponrandom.h"

FixedRGBLayersFromDatasetAndCubeGraphicRepFactory::FixedRGBLayersFromDatasetAndCubeGraphicRepFactory(FixedRGBLayersFromDatasetAndCube *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

FixedRGBLayersFromDatasetAndCubeGraphicRepFactory::~FixedRGBLayersFromDatasetAndCubeGraphicRepFactory() {

}
AbstractGraphicRep* FixedRGBLayersFromDatasetAndCubeGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::View3D || type == ViewType::BasemapView || type == ViewType::StackBasemapView) {
		FixedRGBLayersFromDatasetAndCubeRep *rep = new FixedRGBLayersFromDatasetAndCubeRep(m_data, parent);
		return rep;
	} else if (type == ViewType::InlineView) {
		FixedRGBLayersFromDatasetAndCubeRepOnSlice* rep = nullptr;
		if ( m_data->isInlineXLineDisplay() )
			rep = new FixedRGBLayersFromDatasetAndCubeRepOnSlice(m_data, SliceDirection::Inline, parent);
		return rep;
	} else if (type == ViewType::XLineView) {
		FixedRGBLayersFromDatasetAndCubeRepOnSlice* rep = nullptr;
		if ( m_data->isInlineXLineDisplay() )
			rep = new FixedRGBLayersFromDatasetAndCubeRepOnSlice(m_data, SliceDirection::XLine, parent);
		return rep;
	} else if (type == ViewType::RandomView) {
		FixedRGBLayersFromDatasetAndCubeRepOnRandom* rep = new FixedRGBLayersFromDatasetAndCubeRepOnRandom(
				m_data, parent);
		return rep;
	}
	return nullptr;
}

QList<IGraphicRepFactory*> FixedRGBLayersFromDatasetAndCubeGraphicRepFactory::childReps(ViewType type, AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;

	// IGraphicRepFactory *rep = new FixedRGBLayersFromDatasetAndCubeGraphicRepFactory(m_data);
	// reps.push_back(rep);
	// FixedRGBLayersFromDatasetAndCubeRepOnSlice* rep = new FixedRGBLayersFromDatasetAndCubeRepOnSlice(m_data, SliceDirection::Inline, parent);
	/*
	for (WellBore *d : m_data->wellBores()) {
		IGraphicRepFactory * factory= d->graphicRepFactory();
		reps.push_back(factory);
	}
	*/
	return reps;
}



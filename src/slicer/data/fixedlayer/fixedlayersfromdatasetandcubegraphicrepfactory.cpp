#include "fixedlayersfromdatasetandcubegraphicrepfactory.h"

#include "seismic3dabstractdataset.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "fixedlayersfromdatasetandcube.h"
#include "affine2dtransformation.h"
#include "seismic3dabstractdataset.h"
#include "abstractinnerview.h"
#include "fixedlayersfromdatasetandcuberep.h"
#include "fixedlayersfromdatasetandcubereponslice.h"
#include "fixedlayersfromdatasetandcubereponrandom.h"

FixedLayersFromDatasetAndCubeGraphicRepFactory::FixedLayersFromDatasetAndCubeGraphicRepFactory(FixedLayersFromDatasetAndCube *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

FixedLayersFromDatasetAndCubeGraphicRepFactory::~FixedLayersFromDatasetAndCubeGraphicRepFactory() {

}
AbstractGraphicRep* FixedLayersFromDatasetAndCubeGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::View3D || type == ViewType::BasemapView || type == ViewType::StackBasemapView) {
		FixedLayersFromDatasetAndCubeRep *rep = new FixedLayersFromDatasetAndCubeRep(m_data, parent);
		return rep;
	} else if (type == ViewType::InlineView) {
		FixedLayersFromDatasetAndCubeRepOnSlice* rep = new FixedLayersFromDatasetAndCubeRepOnSlice(m_data, SliceDirection::Inline, parent);
		return rep;
	} else if (type == ViewType::XLineView) {
		FixedLayersFromDatasetAndCubeRepOnSlice* rep = new FixedLayersFromDatasetAndCubeRepOnSlice(m_data, SliceDirection::XLine, parent);
		return rep;
	} else if (type == ViewType::RandomView) {
		FixedLayersFromDatasetAndCubeRepOnRandom* rep = new FixedLayersFromDatasetAndCubeRepOnRandom(
				m_data, parent);
		return rep;
	}
	return nullptr;
}

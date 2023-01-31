#include "fixedlayerfromdatasetgraphicrepfactory.h"

#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "fixedlayerfromdatasetrep.h"
#include "fixedlayerfromdatasetreponslice.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "slicepositioncontroler.h"
#include "cudargbimage.h"
#include "fixedlayerfromdataset.h"
#include "affine2dtransformation.h"
#include "stacklayerrgtrep.h"
#include "fixedlayerfromdatasetreponrandom.h"

FixedLayerFromDatasetGraphicRepFactory::FixedLayerFromDatasetGraphicRepFactory(FixedLayerFromDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

FixedLayerFromDatasetGraphicRepFactory::~FixedLayerFromDatasetGraphicRepFactory() {

}
AbstractGraphicRep* FixedLayerFromDatasetGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type == ViewType::StackBasemapView) {
		FixedLayerFromDatasetRep *rep = new FixedLayerFromDatasetRep(m_data, parent);
		return rep;
	} else if (type == ViewType::InlineView) {
		FixedLayerFromDatasetRepOnSlice* rep = new FixedLayerFromDatasetRepOnSlice(m_data,
				m_data->dataset()->ijToInlineXlineTransfoForInline(), SliceDirection::Inline, parent);
		return rep;
	} else if (type == ViewType::XLineView) {
		FixedLayerFromDatasetRepOnSlice* rep = new FixedLayerFromDatasetRepOnSlice(m_data,
				m_data->dataset()->ijToInlineXlineTransfoForXline(), SliceDirection::XLine, parent);
		return rep;
	} else if (type == ViewType::RandomView) {
		FixedLayerFromDatasetRepOnRandom* rep = new FixedLayerFromDatasetRepOnRandom(m_data, parent);
		return rep;
	} else {
		return nullptr;
	}
}

#include "fixedrgblayersfromdatasetgraphicrepfactory.h"

#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "fixedrgblayersfromdatasetrep.h"
#include "fixedrgblayersfromdatasetreponslice.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "slicepositioncontroler.h"
#include "cudargbimage.h"
#include "fixedrgblayersfromdataset.h"
#include "affine2dtransformation.h"
#include "stacklayerrgtrep.h"
#include "seismic3dabstractdataset.h"
#include "abstractinnerview.h"

FixedRGBLayersFromDatasetGraphicRepFactory::FixedRGBLayersFromDatasetGraphicRepFactory(FixedRGBLayersFromDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

FixedRGBLayersFromDatasetGraphicRepFactory::~FixedRGBLayersFromDatasetGraphicRepFactory() {

}
AbstractGraphicRep* FixedRGBLayersFromDatasetGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::View3D || type == ViewType::BasemapView || type == ViewType::StackBasemapView) {
		FixedRGBLayersFromDatasetRep *rep = new FixedRGBLayersFromDatasetRep(m_data, parent);
		return rep;
	} else if (type == ViewType::InlineView) {
		CUDAImagePaletteHolder *slice = new CUDAImagePaletteHolder(
				m_data->dataset()->width(), m_data->dataset()->height(),
				ImageFormats::QSampleType::INT16,
				m_data->dataset()->ijToInlineXlineTransfoForInline(), parent);
		FixedRGBLayersFromDatasetRepOnSlice* rep = new FixedRGBLayersFromDatasetRepOnSlice(m_data, slice, SliceDirection::Inline, parent);
		return rep;
	} else if (type == ViewType::XLineView) {
		CUDAImagePaletteHolder *slice = new CUDAImagePaletteHolder(
				m_data->dataset()->depth(), m_data->dataset()->height(),
				ImageFormats::QSampleType::INT16,
				m_data->dataset()->ijToInlineXlineTransfoForXline(), parent);
		FixedRGBLayersFromDatasetRepOnSlice* rep = new FixedRGBLayersFromDatasetRepOnSlice(m_data, slice, SliceDirection::XLine, parent);
		return rep;
	}
	return nullptr;
}

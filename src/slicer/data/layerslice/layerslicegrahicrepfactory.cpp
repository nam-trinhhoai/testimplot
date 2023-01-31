#include "layerslicegrahicrepfactory.h"

#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "layerrgtrep.h"
#include "layerrgtreponslice.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "slicepositioncontroler.h"
#include "cudargbimage.h"
#include "LayerSlice.h"
#include "affine2dtransformation.h"
#include "stacklayerrgtrep.h"

LayerSliceGraphicRepFactory::LayerSliceGraphicRepFactory(LayerSlice *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

LayerSliceGraphicRepFactory::~LayerSliceGraphicRepFactory() {

}
AbstractGraphicRep* LayerSliceGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type == ViewType::View3D) {
		LayerRGTRep *rep = new LayerRGTRep(m_data, parent);
		return rep;
	} else if (type==ViewType::StackBasemapView) {
		StackLayerRGTRep *rep = new StackLayerRGTRep(m_data, parent);
		return rep;
	}else if (type == ViewType::InlineView )
	{
		LayerRGTRepOnSlice *rep = new LayerRGTRepOnSlice(m_data, m_data->seismic()->ijToInlineXlineTransfoForInline(),SliceDirection::Inline, parent);
		return rep;
	}else if(type == ViewType::XLineView)
	{
		LayerRGTRepOnSlice *rep = new LayerRGTRepOnSlice(m_data,
				m_data->seismic()->ijToInlineXlineTransfoForXline(),
				SliceDirection::XLine, parent);
		return rep;
	}
	return nullptr;
}

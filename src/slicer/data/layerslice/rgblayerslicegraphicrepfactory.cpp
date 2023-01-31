#include "rgblayerslicegraphicrepfactory.h"

#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "rgblayerrgtrep.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "slicepositioncontroler.h"
#include "cudargbimage.h"
#include "rgblayerslice.h"
#include "affine2dtransformation.h"

RGBLayerSliceGraphicRepFactory::RGBLayerSliceGraphicRepFactory(RGBLayerSlice *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

RGBLayerSliceGraphicRepFactory::~RGBLayerSliceGraphicRepFactory() {

}
AbstractGraphicRep* RGBLayerSliceGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type == ViewType::View3D || type == ViewType::StackBasemapView ) {
		RGBLayerRGTRep *rep = new RGBLayerRGTRep(m_data, parent);
		return rep;
	}
	return nullptr;
}

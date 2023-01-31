

#include "rgblayerfreehorizongraphicrepfactory.h"

#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "rgblayerrgtrep.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "slicepositioncontroler.h"
#include "cudargbimage.h"
#include "rgblayerslice.h"
#include "rgblayerfreehorizongraphicrepfactory.h"
#include <igraphicrepfactory.h>
#include "affine2dtransformation.h"
#include "rgblayerimplfreehorizonslice.h"

#

RGBLayerFreeHorizonGraphicRepFactory::RGBLayerFreeHorizonGraphicRepFactory(RGBLayerImplFreeHorizonOnSlice *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

RGBLayerFreeHorizonGraphicRepFactory::~RGBLayerFreeHorizonGraphicRepFactory() {

}
AbstractGraphicRep* RGBLayerFreeHorizonGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	// if (type == ViewType::BasemapView || type == ViewType::View3D || type == ViewType::StackBasemapView ) {
	RGBLayerImplFreeHorizonOnSlice *rep = new RGBLayerImplFreeHorizonOnSlice(m_data, parent);
		return rep;
	//}
	// return nullptr;
}

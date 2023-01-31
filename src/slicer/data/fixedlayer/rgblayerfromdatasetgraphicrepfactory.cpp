#include "rgblayerfromdatasetgraphicrepfactory.h"
#include "rgblayerfromdataset.h"

#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "rgblayerfromdatasetrep.h"
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

RgbLayerFromDatasetGraphicRepFactory::RgbLayerFromDatasetGraphicRepFactory(RgbLayerFromDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

RgbLayerFromDatasetGraphicRepFactory::~RgbLayerFromDatasetGraphicRepFactory() {

}
AbstractGraphicRep* RgbLayerFromDatasetGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::View3D || type == ViewType::BasemapView || type == ViewType::StackBasemapView ) {
		RgbLayerFromDatasetRep *rep = new RgbLayerFromDatasetRep(m_data, parent);
		return rep;
	}
	return nullptr;
}

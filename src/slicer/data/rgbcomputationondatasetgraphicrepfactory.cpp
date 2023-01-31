#include "rgbcomputationondatasetgraphicrepfactory.h"
#include "rgbcomputationondataset.h"
#include "abstractinnerview.h"

RgbComputationOnDatasetGraphicRepFactory::RgbComputationOnDatasetGraphicRepFactory(
		RgbComputationOnDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

RgbComputationOnDatasetGraphicRepFactory::~RgbComputationOnDatasetGraphicRepFactory() {

}
AbstractGraphicRep* RgbComputationOnDatasetGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	return nullptr;
}

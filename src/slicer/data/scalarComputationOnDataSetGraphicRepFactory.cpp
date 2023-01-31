#include "scalarComputationOnDataSetGraphicRepFactory.h"
#include "scalarComputationOnDataSet.h"
#include "abstractinnerview.h"

ScalarComputationOnDatasetGraphicRepFactory::ScalarComputationOnDatasetGraphicRepFactory(
		ScalarComputationOnDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

ScalarComputationOnDatasetGraphicRepFactory::~ScalarComputationOnDatasetGraphicRepFactory() {

}
AbstractGraphicRep* ScalarComputationOnDatasetGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	return nullptr;
}

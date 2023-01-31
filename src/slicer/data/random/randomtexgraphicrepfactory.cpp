#include "randomtexgraphicrepfactory.h"
#include "randomtexdataset.h"
#include "random3dtexrep.h"

RandomTexGraphicRepFactory::RandomTexGraphicRepFactory(
		RandomTexDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
//	connect(m_data, SIGNAL(datasetAdded(RandomDataset *)), this,SLOT(datasetAdded(RandomDataset *)));

}

RandomTexGraphicRepFactory::~RandomTexGraphicRepFactory() {

}
AbstractGraphicRep* RandomTexGraphicRepFactory::rep(ViewType type,AbstractInnerView *parent) {

	if ( type==ViewType::View3D ) {
			return new Random3dTexRep(m_data, parent);
		}

	return nullptr;
}

QList<IGraphicRepFactory*> RandomTexGraphicRepFactory::childReps(ViewType type,AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;

	return reps;
}


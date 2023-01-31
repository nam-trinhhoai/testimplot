#include "randomgraphicrepfactory.h"
#include "randomdataset.h"
#include "random3drep.h"

RandomGraphicRepFactory::RandomGraphicRepFactory(
		RandomDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
	connect(m_data, SIGNAL(datasetAdded(RandomTexDataset *)), this,
			SLOT(datasetAdded(RandomTexDataset *)));

}

RandomGraphicRepFactory::~RandomGraphicRepFactory() {

}
AbstractGraphicRep* RandomGraphicRepFactory::rep(ViewType type,AbstractInnerView *parent) {

	if ( type==ViewType::View3D ) {
			return new Random3dRep(m_data, parent);
		}

	return nullptr;
}

QList<IGraphicRepFactory*> RandomGraphicRepFactory::childReps(ViewType type,AbstractInnerView *parent) {


	QList<IGraphicRepFactory*> reps;
	for (RandomTexDataset *d : m_data->datasets()) {
		IGraphicRepFactory * factory= d->graphicRepFactory();
		reps.push_back(factory);
	}
	return reps;
}

void RandomGraphicRepFactory::datasetAdded(RandomTexDataset *dataset)
{
	qDebug()<<" random add data set : "<<dataset->name();
	emit childAdded(dataset->graphicRepFactory());
}

void RandomGraphicRepFactory::datasetRemoved(RandomTexDataset *dataset)
{
	emit childRemoved(dataset->graphicRepFactory());
}


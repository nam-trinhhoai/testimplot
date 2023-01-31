#include "randomdataset.h"

#include "randomgraphicrepfactory.h"
RandomDataset::RandomDataset(WorkingSetManager * workingSet,RandomView3D* random,const QString &name, QObject *parent) :
		IData(workingSet, parent), m_name(name) {

	m_uuid = QUuid::createUuid();
	m_repFactory = new RandomGraphicRepFactory(this);
	m_random=random;
}

RandomDataset::~RandomDataset() {

}

//IData
IGraphicRepFactory *RandomDataset::graphicRepFactory() {
	return m_repFactory;
}

QUuid RandomDataset::dataID() const {
	return m_uuid;
}

RandomView3D* RandomDataset::getRandom3d()
{
	return m_random;
}

void RandomDataset::addDataset(RandomTexDataset *dataset) {

	m_datasets.push_back(dataset);
	emit datasetAdded(dataset);

}

void RandomDataset::removeDataset(RandomTexDataset *dataset) {
	m_datasets.removeOne(dataset);
	emit datasetRemoved(dataset);

}

QList<RandomTexDataset*> RandomDataset::datasets() {
	return m_datasets;
}

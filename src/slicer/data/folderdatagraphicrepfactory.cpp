#include "folderdatagraphicrepfactory.h"
#include "folderdata.h"
#include "folderdatarep.h"

FolderDataGraphicRepFactory::FolderDataGraphicRepFactory(
		FolderData *data) :
		IGraphicRepFactory(data) {
	m_data = data;
	connect(m_data, SIGNAL(dataAdded(IData *)), this,
			SLOT(dataAdded(IData *)));
	connect(m_data, SIGNAL(dataRemoved(IData *)), this,
			SLOT(dataRemoved(IData *)));
}

FolderDataGraphicRepFactory::~FolderDataGraphicRepFactory() {

}
AbstractGraphicRep* FolderDataGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	return new FolderDataRep(m_data, parent);
}

QList<IGraphicRepFactory*> FolderDataGraphicRepFactory::childReps(ViewType type,
		AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;
	for (IData *d : m_data->data()) {
		IGraphicRepFactory * factory= d->graphicRepFactory();
		reps.push_back(factory);
	}
	return reps;
}

void FolderDataGraphicRepFactory::dataAdded(
		IData *data) {
	emit childAdded(data->graphicRepFactory());
}

void FolderDataGraphicRepFactory::dataRemoved(
		IData *data) {
	emit childRemoved(data->graphicRepFactory());
}

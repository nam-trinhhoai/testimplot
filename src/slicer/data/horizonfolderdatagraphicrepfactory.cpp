#include "horizonfolderdatagraphicrepfactory.h"
#include "horizonfolderdata.h"
#include "horizondatarep.h"
#include "horizonfolderreponslice.h"
#include "horizonfolderreponrandom.h"
#include "sliceutils.h"
HorizonFolderDataGraphicRepFactory::HorizonFolderDataGraphicRepFactory(
		HorizonFolderData *data) :
		IGraphicRepFactory(data) {
	m_data = data;
//	connect(m_data, SIGNAL(dataAdded(IData *)), this,SLOT(dataAdded(IData *)));
//	connect(m_data, SIGNAL(dataRemoved(IData *)), this,SLOT(dataRemoved(IData *)));
}

HorizonFolderDataGraphicRepFactory::~HorizonFolderDataGraphicRepFactory() {

}
AbstractGraphicRep* HorizonFolderDataGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {

	if(type == View3D)return new HorizonDataRep(m_data, parent);

	if(type == RandomView ) return new HorizonFolderRepOnRandom( m_data,parent);

	if(type == XLineView ) return new HorizonFolderRepOnSlice(m_data,SliceDirection::XLine ,parent);
	if(type == InlineView )return new HorizonFolderRepOnSlice(m_data,SliceDirection::Inline,parent);


	return nullptr;
}
/*
QList<IGraphicRepFactory*> HorizonFolderDataGraphicRepFactory::childReps(ViewType type,
		AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;
	for (IData *d : m_data->data()) {
		IGraphicRepFactory * factory= d->graphicRepFactory();
		reps.push_back(factory);
	}
	return reps;
}

void HorizonFolderDataGraphicRepFactory::dataAdded(
		IData *data) {
	emit childAdded(data->graphicRepFactory());
}

void HorizonFolderDataGraphicRepFactory::dataRemoved(
		IData *data) {
	emit childRemoved(data->graphicRepFactory());
}
*/

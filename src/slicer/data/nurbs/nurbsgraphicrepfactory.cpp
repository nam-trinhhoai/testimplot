#include "nurbsgraphicrepfactory.h"
#include "nurbsdataset.h"
//#include "wellpick.h"
#include "nurbsrep.h"

NurbsGraphicRepFactory::NurbsGraphicRepFactory(
		NurbsDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
//	connect(m_data, SIGNAL(wellPickAdded(WellPick*)), this,SLOT(wellPickAdded(WellPick*)));
}

NurbsGraphicRepFactory::~NurbsGraphicRepFactory() {

}
AbstractGraphicRep* NurbsGraphicRepFactory::rep(ViewType type,AbstractInnerView *parent) {
	/*if (type == ViewType::InlineView || type == ViewType::XLineView || type==ViewType::View3D || type==ViewType::RandomView) {
		return new MarkerRep(m_data, parent);
	}*/

	if ( type==ViewType::View3D ) {
			return new NurbsRep(m_data, parent);
		}

	return nullptr;
}

QList<IGraphicRepFactory*> NurbsGraphicRepFactory::childReps(ViewType type,AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;
	/*for (WellPick *d : m_data->wellPicks()) {
		IGraphicRepFactory * factory= d->graphicRepFactory();
		reps.push_back(factory);
	}*/
	return reps;
}

/*
void NurbsGraphicRepFactory::wellPickAdded(WellPick* wellPick) {
	emit childAdded(wellPick->graphicRepFactory());
}

void NurbsGraphicRepFactory::wellPickRemoved(
		WellPick* wellPick) {
	emit childRemoved(wellPick->graphicRepFactory());
}*/

#include "markergraphicrepfactory.h"
#include "marker.h"
#include "wellpick.h"
#include "markerrep.h"

MarkerGraphicRepFactory::MarkerGraphicRepFactory(
		Marker *data) :
		IGraphicRepFactory(data) {
	m_data = data;
	connect(m_data, SIGNAL(wellPickAdded(WellPick*)), this,
			SLOT(wellPickAdded(WellPick*)));
	connect(m_data, SIGNAL(wellPickRemoved(WellPick*)), this,
			SLOT(wellPickRemoved(WellPick*)));
}

MarkerGraphicRepFactory::~MarkerGraphicRepFactory() {

}
AbstractGraphicRep* MarkerGraphicRepFactory::rep(ViewType type,AbstractInnerView *parent) {
	if (type == ViewType::InlineView || type == ViewType::XLineView || type==ViewType::View3D || type==ViewType::RandomView) {
		return new MarkerRep(m_data, parent);
	}
//	if (type == ViewType::BasemapView || type==ViewType::StackBasemapView)
//		return new WellBoreRepOnMap(m_data, parent);
//	else if (type == ViewType::InlineView || type == ViewType::XLineView) {
//		SliceDirection dir = (type==ViewType::InlineView) ? SliceDirection::Inline : SliceDirection::XLine;
//		return new WellBoreRepOnSlice(m_data, dir, parent);
//	}
//	} else if (type == ViewType::View3D) {
//		return new SeismicSurveyRepOn3D(m_data, parent);
//	}

	return nullptr;
}

QList<IGraphicRepFactory*> MarkerGraphicRepFactory::childReps(ViewType type,AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;
	for (WellPick *d : m_data->wellPicks()) {
		IGraphicRepFactory * factory= d->graphicRepFactory();
		reps.push_back(factory);
	}
	return reps;
}

void MarkerGraphicRepFactory::wellPickAdded(WellPick* wellPick) {
	emit childAdded(wellPick->graphicRepFactory());
}

void MarkerGraphicRepFactory::wellPickRemoved(
		WellPick* wellPick) {
	emit childRemoved(wellPick->graphicRepFactory());
}

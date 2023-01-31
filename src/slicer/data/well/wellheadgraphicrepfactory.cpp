#include "wellheadgraphicrepfactory.h"
#include "wellhead.h"
#include "wellbore.h"
#include "wellheadreponmap.h"
#include "wellheadreponslice.h"
#include "wellheadlayeronslice.h"
#include "wellheadrepnodisplay.h"

WellHeadGraphicRepFactory::WellHeadGraphicRepFactory(
		WellHead *data) :
		IGraphicRepFactory(data) {
	m_data = data;
	connect(m_data, SIGNAL(wellBoreAdded(WellBore *)), this,
			SLOT(wellBoreAdded(WellBore *)));
}

WellHeadGraphicRepFactory::~WellHeadGraphicRepFactory() {

}
AbstractGraphicRep* WellHeadGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type==ViewType::StackBasemapView) {
		return new WellHeadRepOnMap(m_data, parent);
	} else if (type == ViewType::View3D) {
		return new WellHeadRepNoDisplay(m_data, parent);
	}
//	if (type == ViewType::BasemapView || type == ViewType::StackBasemapView)
//		return new SeismicSurveyRepOnMap(m_data, parent);
	else if (type == ViewType::InlineView || type == ViewType::XLineView) {
		return new WellHeadRepOnSlice(m_data, parent);
//	} else if (type == ViewType::View3D) {
//		return new SeismicSurveyRepOn3D(m_data, parent);
	} else if (type == ViewType::RandomView) {
		return new WellHeadRepOnSlice(m_data, parent);
	}

	return nullptr;
}

QList<IGraphicRepFactory*> WellHeadGraphicRepFactory::childReps(ViewType type,
		AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;
	for (WellBore *d : m_data->wellBores()) {
		IGraphicRepFactory * factory= d->graphicRepFactory();
		reps.push_back(factory);
	}
	return reps;
}

void WellHeadGraphicRepFactory::wellBoreAdded(
		WellBore *wellBore) {
	emit childAdded(wellBore->graphicRepFactory());
}

void WellHeadGraphicRepFactory::wellBoreRemoved(
		WellBore *wellBore) {
	emit childRemoved(wellBore->graphicRepFactory());
}

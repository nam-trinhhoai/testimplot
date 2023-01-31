#include "wellpickgraphicrepfactory.h"
#include "wellpick.h"
#include "wellpickreponslice.h"
#include "wellpickrep.h"
#include "wellpickreponrandom.h"

WellPickGraphicRepFactory::WellPickGraphicRepFactory(
		WellPick *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

WellPickGraphicRepFactory::~WellPickGraphicRepFactory() {

}
AbstractGraphicRep* WellPickGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::InlineView || type == ViewType::XLineView) {
		return new WellPickRepOnSlice(m_data, parent);
	} else if (type == ViewType::View3D) {
		return new WellPickRep(m_data, parent);
	} else if (type == ViewType::RandomView) {
		return new WellPickRepOnRandom(m_data, parent);
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

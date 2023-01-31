#include "wellboregraphicrepfactory.h"
#include "wellbore.h"
#include "wellborereponslice.h"
#include "wellborereponmap.h"
#include "wellborerepon3d.h"
#include "wellborereponrandom.h"
//#include "seismicsurveyrepon3D.h"

WellBoreGraphicRepFactory::WellBoreGraphicRepFactory(
		WellBore *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

WellBoreGraphicRepFactory::~WellBoreGraphicRepFactory() {

}
AbstractGraphicRep* WellBoreGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type==ViewType::StackBasemapView)
		return new WellBoreRepOnMap(m_data, parent);
	else if (type == ViewType::InlineView || type == ViewType::XLineView) {
		SliceDirection dir = (type==ViewType::InlineView) ? SliceDirection::Inline : SliceDirection::XLine;
		return new WellBoreRepOnSlice(m_data, dir, parent);
	} else if (type == ViewType::RandomView) {
		return new WellBoreRepOnRandom(m_data, parent);
	} else if (type==ViewType::View3D) {
		return new WellBoreRepOn3D(m_data, parent);
	}
//	} else if (type == ViewType::View3D) {
//		return new SeismicSurveyRepOn3D(m_data, parent);
//	}

	return nullptr;
}

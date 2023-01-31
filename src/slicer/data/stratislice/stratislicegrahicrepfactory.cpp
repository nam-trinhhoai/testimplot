#include "stratislicegrahicrepfactory.h"
#include "stratislicerep.h"
#include "frequencystratisliceattribute.h"
#include "amplitudestratisliceattribute.h"
#include "rgbstratisliceattribute.h"
#include "stratislice.h"

StratiSliceGraphicRepFactory::StratiSliceGraphicRepFactory(StratiSlice *data) :
		IGraphicRepFactory() {
	m_data = data;
}

StratiSliceGraphicRepFactory::~StratiSliceGraphicRepFactory() {

}

QList<IGraphicRepFactory*> StratiSliceGraphicRepFactory::childReps(ViewType type,
		AbstractInnerView *parent) {
	QList<IGraphicRepFactory*> reps;
	reps.push_back(m_data->frequencyAttribute()->graphicRepFactory());
	reps.push_back(m_data->rgbAttribute()->graphicRepFactory());
	reps.push_back(m_data->amplitudeAttribute()->graphicRepFactory());
	return reps;
}

AbstractGraphicRep* StratiSliceGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type == ViewType::StackBasemapView 
			|| type == ViewType::View3D || type == ViewType::InlineView 
			|| type == ViewType::XLineView) {
		StratiSliceRep *rep = new StratiSliceRep(m_data, parent);
		return rep;
	}
	return nullptr;
}

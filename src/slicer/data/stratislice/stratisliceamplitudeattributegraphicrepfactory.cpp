#include "stratisliceamplitudeattributegraphicrepfactory.h"
#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "slicepositioncontroler.h"
#include "cudargbimage.h"
#include "seismicsurvey.h"
#include "affine2dtransformation.h"
#include "stratisliceamplituderep.h"
#include "stratislice.h"
#include "stratisliceattributereponslice.h"
#include "amplitudestratisliceattribute.h"

StratiSliceAmplitudeAttributeGraphicRepFactory::StratiSliceAmplitudeAttributeGraphicRepFactory(
		AmplitudeStratiSliceAttribute *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

StratiSliceAmplitudeAttributeGraphicRepFactory::~StratiSliceAmplitudeAttributeGraphicRepFactory() {

}

AbstractGraphicRep* StratiSliceAmplitudeAttributeGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type == ViewType::StackBasemapView || type == ViewType::View3D ) {
		StratiSliceAmplitudeRep * rep=new StratiSliceAmplitudeRep(m_data, parent);
		return rep;
	}else if (type == ViewType::InlineView )
	{
		StratiSliceAttributeRepOnSlice *rep = new StratiSliceAttributeRepOnSlice(m_data, m_data->stratiSlice()->seismic()->ijToInlineXlineTransfoForInline(),SliceDirection::Inline, parent);
		return rep;
	}else if(type == ViewType::XLineView)
	{
		StratiSliceAttributeRepOnSlice *rep = new StratiSliceAttributeRepOnSlice(m_data,m_data->stratiSlice()->seismic()->ijToInlineXlineTransfoForXline(),SliceDirection::XLine, parent);
		return rep;
	}
	return nullptr;
}

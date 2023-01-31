#include "stratislicefrequencyattributegraphicrepfactory.h"
#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"
#include "slicepositioncontroler.h"
#include "cudargbimage.h"
#include "seismicsurvey.h"
#include "affine2dtransformation.h"
#include "stratislicefrequencyrep.h"
#include "stratislice.h"
#include "stratisliceattributereponslice.h"
#include "frequencystratisliceattribute.h"

StratiSliceFrequencyAttributeGraphicRepFactory::StratiSliceFrequencyAttributeGraphicRepFactory(
		FrequencyStratiSliceAttribute *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

StratiSliceFrequencyAttributeGraphicRepFactory::~StratiSliceFrequencyAttributeGraphicRepFactory() {

}

AbstractGraphicRep* StratiSliceFrequencyAttributeGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type == ViewType::StackBasemapView || type == ViewType::View3D ) {
		StratiSliceFrequencyRep * rep=new StratiSliceFrequencyRep(m_data, parent);
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

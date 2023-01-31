#include "multiseedhorizongraphicrepfactory.h"
#include "abstractinnerview.h"
#include "multiseedhorizon.h"
#include "cudaimagepaletteholder.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"

#include "multiseedslicerep.h"
#include "multiseedrandomrep.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"


MultiSeedHorizonGraphicRepFactory::MultiSeedHorizonGraphicRepFactory(
		MultiSeedHorizon *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

MultiSeedHorizonGraphicRepFactory::~MultiSeedHorizonGraphicRepFactory() {

}

AbstractGraphicRep* MultiSeedHorizonGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	Seismic3DAbstractDataset* seismicData = m_data->seismic();
	if (type == ViewType::InlineView )
	{
		std::array<double, 6> transfo = seismicData->ijToInlineXlineTransfoForInline()->direct();
		QPair<QVector2D, AffineTransformation> rangeAndStep(
						QVector2D(transfo[3],
										transfo[3] + transfo[5] * (seismicData->depth() - 1)),
						AffineTransformation(transfo[5], transfo[3]));
		MultiSeedSliceRep *rep = new MultiSeedSliceRep(m_data, rangeAndStep,SliceDirection::Inline, parent);
		return rep;
	}else if(type == ViewType::XLineView)
	{
		std::array<double, 6> transfo = seismicData->ijToInlineXlineTransfoForXline()->direct();
		QPair<QVector2D, AffineTransformation> rangeAndStep(
						QVector2D(transfo[0],
										transfo[0] + transfo[1] * (seismicData->width() - 1)),
						AffineTransformation(transfo[1], transfo[0]));
		MultiSeedSliceRep *rep = new MultiSeedSliceRep(m_data, rangeAndStep,SliceDirection::XLine, parent);
		return rep;
	}else if(type == ViewType::RandomView)
	{
		MultiSeedRandomRep *rep = new MultiSeedRandomRep(m_data, parent);
		return rep;
	}
	return nullptr;
}


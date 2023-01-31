#include "computationoperatordatasetgraphicrepfactory.h"
#include "computationoperatordataset.h"
#include "abstractinnerview.h"
#include "computationoperatordatasetrep.h"
#include "cudaimagepaletteholder.h"
#include "affine2dtransformation.h"

#include <QPair>

ComputationOperatorDatasetGraphicRepFactory::ComputationOperatorDatasetGraphicRepFactory(
		ComputationOperatorDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

ComputationOperatorDatasetGraphicRepFactory::~ComputationOperatorDatasetGraphicRepFactory() {

}
AbstractGraphicRep* ComputationOperatorDatasetGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type==ViewType::InlineView) {
		CUDAImagePaletteHolder *slice = new CUDAImagePaletteHolder(
				m_data->width(), m_data->height(),
				m_data->sampleType(),
				m_data->ijToInlineXlineTransfoForInline(), parent);
		slice->setLookupTable(m_data->defaultLookupTable());

		std::array<double, 6> transfo =
				m_data->ijToInlineXlineTransfo()->direct();
		QPair<QVector2D, AffineTransformation> rangeAndStep(
				QVector2D(transfo[3],
						transfo[3] + transfo[5] * (m_data->depth() - 1)),
				AffineTransformation(transfo[5], transfo[3]));
		ComputationOperatorDatasetRep *rep = new ComputationOperatorDatasetRep(m_data, slice, rangeAndStep,
				SliceDirection::Inline, parent);

		return rep;
	}
	else if (type==ViewType::XLineView) {
		CUDAImagePaletteHolder *slice = new CUDAImagePaletteHolder(
				m_data->depth(), m_data->height(),
				m_data->sampleType(),
				m_data->ijToInlineXlineTransfoForXline(), parent);
		slice->setLookupTable(m_data->defaultLookupTable());

		std::array<double, 6> transfo =
						m_data->ijToInlineXlineTransfo()->direct();
		QPair<QVector2D, AffineTransformation> rangeAndStep(
				QVector2D(transfo[0],
						transfo[0] + transfo[1] * (m_data->width() - 1)),
						AffineTransformation(transfo[1], transfo[0]));

		ComputationOperatorDatasetRep *rep = new ComputationOperatorDatasetRep(m_data, slice, rangeAndStep,
				SliceDirection::XLine, parent);

		return rep;
	}
	return nullptr;
}

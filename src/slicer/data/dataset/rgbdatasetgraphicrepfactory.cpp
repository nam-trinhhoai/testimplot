#include "rgbdatasetgraphicrepfactory.h"
#include "rgbdataset.h"
#include "rgbdatasetreponslice.h"
#include "rgbdatasetreponrandom.h"

#include "seismic3dabstractdataset.h"
#include "abstractinnerview.h"
#include "cudargbimage.h"
#include "cudaimagepaletteholder.h"
#include "affinetransformation.h"

#include <QPair>
#include <QVector2D>

RgbDatasetGraphicRepFactory::RgbDatasetGraphicRepFactory(
		RgbDataset *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

RgbDatasetGraphicRepFactory::~RgbDatasetGraphicRepFactory() {

}

AbstractGraphicRep* RgbDatasetGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type==ViewType::InlineView || type==ViewType::XLineView) {
		SliceDirection dir;
		const Affine2DTransformation* transfo;
		std::array<double, 6> transfoArray =
						m_data->ijToInlineXlineTransfo()->direct();
		QPair<QVector2D, AffineTransformation> rangeAndStep;
		if (type==ViewType::InlineView) {
			dir = SliceDirection::Inline;
			transfo = m_data->ijToInlineXlineTransfoForInline();
			rangeAndStep = QPair<QVector2D, AffineTransformation>(
					QVector2D(transfoArray[3],
							transfoArray[3] + transfoArray[5] * (m_data->depth() - 1)),
					AffineTransformation(transfoArray[5], transfoArray[3]));
		} else {
			dir = SliceDirection::XLine;
			transfo = m_data->ijToInlineXlineTransfoForXline();
			rangeAndStep = QPair<QVector2D, AffineTransformation>(
					QVector2D(transfoArray[0],
							transfoArray[0] + transfoArray[1] * (m_data->width() - 1)),
					AffineTransformation(transfoArray[1], transfoArray[0]));
		}
		CUDAImagePaletteHolder *red = new CUDAImagePaletteHolder(
				m_data->width(), m_data->height(),
				m_data->red()->sampleType(),
				transfo, parent);
		CUDAImagePaletteHolder *green = new CUDAImagePaletteHolder(
				m_data->width(), m_data->height(),
				m_data->green()->sampleType(),
				transfo, parent);
		CUDAImagePaletteHolder *blue = new CUDAImagePaletteHolder(
				m_data->width(), m_data->height(),
				m_data->blue()->sampleType(),
				transfo, parent);

		CUDAImagePaletteHolder *alpha = nullptr;
		if (m_data->alpha()!=nullptr) {
			alpha = new CUDAImagePaletteHolder(
					m_data->width(), m_data->height(),
					m_data->alpha()->sampleType(),
					transfo, parent);
		}

		RgbDatasetRepOnSlice *rep = new RgbDatasetRepOnSlice(m_data, red,
				green, blue, alpha, rangeAndStep, dir, parent);

		return rep;
	} else if (type==ViewType::RandomView) {
		return new RgbDatasetRepOnRandom(m_data, parent);
	}
	return nullptr;
}


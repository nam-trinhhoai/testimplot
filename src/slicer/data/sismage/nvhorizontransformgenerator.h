#ifndef NvHorizonTransformGenerator_H
#define NvHorizonTransformGenerator_H

#include <string>
#include "affine2dtransformation.h"
#include "affinetransformation.h"


class NvHorizonTransformGenerator {
public:
	// dimVHint to keep coherency with Seismic3DAbstractDataset loadFromXt
	NvHorizonTransformGenerator(const std::string& horizonPath); // for datasets
	virtual ~NvHorizonTransformGenerator();

	Affine2DTransformation inlineXlineTransfo() const;

	Affine2DTransformation inlineXlineTransfoForInline() const;
	Affine2DTransformation inlineXlineTransfoForXline() const;

	AffineTransformation sampleTransfo() const;

private:
	double m_firstDsInline = 0;
	int m_inlineDsDim = 1;

	double m_inlineDsStep = 1;
	double m_firstDsXline = 0;
	int m_xlineDsDim = 1;

	double m_xlineDsStep = 1;

	double m_firstDsSample = 0;
	int m_sampleDsDim = 1;
	double m_sampleDsStep = 1;
};

#endif

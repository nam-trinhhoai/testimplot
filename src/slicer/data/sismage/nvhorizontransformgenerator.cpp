#include "nvhorizontransformgenerator.h"
#include "Xt.h"

NvHorizonTransformGenerator::NvHorizonTransformGenerator(const std::string & horizonPath) {
	inri::Xt xt(horizonPath);

	m_firstDsInline = xt.startSlice();
	m_inlineDsDim = xt.nRecords();
	m_inlineDsStep = xt.stepSlices();

	m_firstDsXline = xt.startRecord();
	m_xlineDsDim = xt.nSamples();
	m_xlineDsStep =  xt.stepRecords();

	m_firstDsSample = xt.startSamples();
	m_sampleDsDim = xt.nSlices();
	m_sampleDsStep =  xt.stepSamples();
}

Affine2DTransformation NvHorizonTransformGenerator::inlineXlineTransfo() const
{
	std::array<double, 6> result;

	result[0]=m_firstDsXline;
	result[1]=m_xlineDsStep;
	result[2]=0;

	result[3]=m_firstDsInline;
	result[4]=0;
	result[5]=m_inlineDsStep;

	return Affine2DTransformation(m_xlineDsDim,m_inlineDsDim,result);
}

Affine2DTransformation NvHorizonTransformGenerator::inlineXlineTransfoForInline() const
{
	std::array<double, 6> result;
	result[0]=m_firstDsXline;
	result[1]=m_xlineDsStep;
	result[2]=0;

	result[3]=m_firstDsSample;
	result[4]=0;
	result[5]=m_sampleDsStep;

	return Affine2DTransformation(m_xlineDsDim,m_sampleDsDim,result);
}

Affine2DTransformation NvHorizonTransformGenerator::inlineXlineTransfoForXline() const
{
	std::array<double, 6> result;
	result[0]=m_firstDsInline;
	result[1]=m_inlineDsStep;
	result[2]=0;

	result[3]=m_firstDsSample;
	result[4]=0;
	result[5]=m_sampleDsStep;

	return Affine2DTransformation(m_inlineDsDim,m_sampleDsDim,result);
}

AffineTransformation NvHorizonTransformGenerator::sampleTransfo() const
{
	return AffineTransformation(m_sampleDsStep,m_firstDsSample);
}

NvHorizonTransformGenerator::~NvHorizonTransformGenerator() {

}


#include "smdataset3D.h"
#include "Xt.h"

#include "sismagedbmanager.h"
#include "smsurvey3D.h"

SmDataset3D::SmDataset3D(const std::string & datasetPath, int dimVHint) /*:
		m_survey3D(SmSurvey3D(SismageDBManager::survey3DPathFromDatasetPath(datasetPath))) /* dead code*/ {
	// Dataset3D
	inri::Xt xt(datasetPath);

	m_firstDsInline = xt.startSlice();
	m_inlineDsDim = xt.nSlices();
	m_inlineDsStep = xt.stepSlices();

	// Dead code
	//m_inlineDsDist = (m_inlineDsStep / m_survey3D.inlineStep()) *
	//		m_survey3D.inlineDist();

	m_firstDsXline = xt.startRecord();
	m_xlineDsDim = xt.nRecords();
	m_xlineDsStep =  xt.stepRecords();
	// Dead code
	//m_xlineDsDist = (m_xlineDsStep / m_survey3D.xlineStep()) *
	//		m_survey3D.xlineDist();

	if (dimVHint>0 && xt.pixel_dimensions()==1 && xt.nSamples()%dimVHint==0) {
		m_sampleDsDim = xt.nSamples() / dimVHint;
		m_dimV = dimVHint;
	} else {
		m_sampleDsDim = xt.nSamples();
		m_dimV = xt.pixel_dimensions();
	}
	m_firstDsSample = xt.startSamples();
	m_sampleDsDim = xt.nSamples();
	m_sampleDsStep =  xt.stepSamples();
}

Affine2DTransformation SmDataset3D::inlineXlineTransfo() const
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

Affine2DTransformation SmDataset3D::inlineXlineTransfoForInline() const
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

Affine2DTransformation SmDataset3D::inlineXlineTransfoForXline() const
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

AffineTransformation SmDataset3D::sampleTransfo() const
{
	return AffineTransformation(m_sampleDsStep,m_firstDsSample);
}

int SmDataset3D::dimV() const {
	return m_dimV;
}

SmDataset3D::~SmDataset3D() {

}


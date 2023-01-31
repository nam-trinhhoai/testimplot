#ifndef SmDataset3D_H
#define SmDataset3D_H

#include <string>
#include "smsurvey3D.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"


class SmDataset3D {
public:
	// dimVHint to keep coherency with Seismic3DAbstractDataset loadFromXt
	SmDataset3D(const std::string &datasetPath, int dimVHint=-1); // for datasets
	virtual ~SmDataset3D();

	Affine2DTransformation inlineXlineTransfo() const;

	Affine2DTransformation inlineXlineTransfoForInline() const;
	Affine2DTransformation inlineXlineTransfoForXline() const;

	AffineTransformation sampleTransfo() const;

	int dimV() const;

private:
	double m_firstDsInline = 0;
	int m_inlineDsDim = 1;

	double m_inlineDsStep = 1;
	// dead code
	//double m_inlineDsDist = 1;
	double m_firstDsXline = 0;
	int m_xlineDsDim = 1;

	double m_xlineDsStep = 1;
	// dead code
	//double m_xlineDsDist = 1;

	double m_firstDsSample = 0;
	int m_sampleDsDim = 1;
	double m_sampleDsStep = 1;

	// Dead code
	//SmSurvey3D m_survey3D;
	int m_dimV; // only saved to allow feedback from dimVHint
};

#endif

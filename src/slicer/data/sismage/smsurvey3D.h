#ifndef SmSurvey3D_H
#define SmSurvey3D_H


#include <string>
#include "affine2dtransformation.h"

class SmSurvey3D {
public:
	SmSurvey3D(const std::string &surveyPath);
	virtual ~SmSurvey3D();

	double firstInline() const {
		return m_firstInline;
	}

	double firstXline() const {
		return m_firstXline;
	}

	int inlineDim() const {
		return m_inlineDim;
	}

	double inlineStep() const {
		return m_inlineStep;
	}

	int xlineDim() const {
		return m_xlineDim;
	}

	double xlineStep() const {
		return m_xlineStep;
	}

	double inlineDist() const {
		return m_inlineDist;
	}

	double xlineDist() const {
		return m_xlineDist;
	}

	Affine2DTransformation inlineXlineToXYTransfo() ;

	Affine2DTransformation ijToXYTransfo() ;

	bool isValid() const;


//	bool checkTransformation();
private:
	void computeTransformations();
private:
	std::string  m_surveyPath;

	double m_firstInline = 0;
	int m_inlineDim = 1;
	double m_inlineStep = 1;
	double m_inlineDist = 1;
	double m_firstXline = 0;
	int m_xlineDim = 1;
	double m_xlineStep = 1;
	double m_xlineDist = 1;

	Affine2DTransformation * m_inlineToXYTransfo;
	Affine2DTransformation *m_ijToXYTransfo;

	bool m_valid = false;

};
#endif

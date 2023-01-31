/*
 * CubeSeismicAddon.h
 *
 *  Created on: 12 mars 2018
 *      Author: j0334308
 */

#ifndef MURATBASICIO_SRC_CubeSeismicAddon_H_
#define MURATBASICIO_SRC_CubeSeismicAddon_H_

#include "viewutils.h"

/** Cube Dimensions. Defines the size of the cube along each i, j and k axis. */
class CubeSeismicAddon {
private:
	float m_firstSample;
	float m_sampleStep;
	float m_firstXline;
	float m_xlineStep;
	float m_firtInline;
	float m_inlineStep;
	double m_inlineAngle;
	double m_xlineAngle;
	SampleUnit m_sampleUnit;

public:
	CubeSeismicAddon():
		m_firstSample(0), m_sampleStep(1),
		m_firstXline(0), m_xlineStep(1), m_firtInline(0),
		m_inlineStep(1), m_inlineAngle(0), m_xlineAngle(90),
		m_sampleUnit(SampleUnit::TIME) {}

	CubeSeismicAddon(float firstSample, float sampleStep,
			float firstXline, float xlineStep, float firtInline,
			float inlineStep, SampleUnit sampleUnit=SampleUnit::TIME):
		m_firstSample(firstSample), m_sampleStep(sampleStep),
		m_firstXline(firstXline), m_xlineStep(xlineStep),
		m_firtInline(firtInline), m_inlineStep(inlineStep),
		m_inlineAngle(0), m_xlineAngle(90), m_sampleUnit(sampleUnit) {}

	float getFirstXline() const {
		return m_firstXline;
	}

	void setFirstXline(float firstXline) {
		m_firstXline = firstXline;
	}

	float getFirstInline() const {
		return m_firtInline;
	}

	void setFirstInline(float firtInline) {
		m_firtInline = firtInline;
	}

	float getInlineStep() const {
		return m_inlineStep;
	}

	void setInlineStep(float inlineStep) {
		m_inlineStep = inlineStep;
	}

	float getXlineStep() const {
		return m_xlineStep;
	}

	void setXlineStep(float xlineStep) {
		m_xlineStep = xlineStep;
	}

	void set(float firstSample, float sampleStep,
			float firstXline, float xlineStep, float firstInline, float inlineStep) {
		m_firstSample = firstSample;
		m_sampleStep = sampleStep;
		m_firstXline = firstXline;
		m_xlineStep = xlineStep;
		m_firtInline = firstInline;
		m_inlineStep = inlineStep;
	}

	double getInlineAngle() const {
		return m_inlineAngle;
	}

	void setInlineAngle(double inlineAngle) {
		m_inlineAngle = inlineAngle;
	}

	double getXlineAngle() const {
		return m_xlineAngle;
	}

	void setXlineAngle(double xlineAngle) {
		m_xlineAngle = xlineAngle;
	}

	float getFirstSample() const {
		return m_firstSample;
	}

	void setFirstSample(float firstSample) {
		m_firstSample = firstSample;
	}

	float getSampleStep() const {
		return m_sampleStep;
	}

	void setSampleStep(float sampleStep) {
		m_sampleStep = sampleStep;
	}

	SampleUnit getSampleUnit() const {
		return m_sampleUnit;
	}

	void setSampleUnit(SampleUnit sampleUnit) {
		m_sampleUnit = sampleUnit;
	}

	bool compare3DGrid(const CubeSeismicAddon& o) {
		return m_inlineAngle==o.m_inlineAngle && m_xlineAngle==o.m_xlineAngle && m_firstSample==o.m_firstSample &&
				m_sampleStep==o.m_sampleStep && m_firstXline==o.m_firstXline && m_firtInline==o.m_firtInline &&
				m_inlineStep==o.m_inlineStep && m_xlineStep==o.m_xlineStep;
	}
};

#endif /* MURATBASICIO_SRC_CubeSeismicAddon_H_ */

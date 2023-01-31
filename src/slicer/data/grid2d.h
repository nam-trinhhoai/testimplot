#ifndef SRC_SLICER_DATA_GRID2D_H
#define SRC_SLICER_DATA_GRID2D_H

#include "affine2dtransformation.h"
#include "viewutils.h"

#include <string>
#include <memory>

class Grid2D {
public:
	Grid2D();
	Grid2D(const Grid2D& other);
	Grid2D(double startInline, double startXLine, double stepInline, double stepXLine,
			long countInline, long countXLine, SampleUnit depthAxis,
			const Affine2DTransformation& inlineXlineToXY);
	Grid2D(double startInline, double startXLine, double stepInline, double stepXLine,
			long countInline, long countXLine, SampleUnit depthAxis,
			std::shared_ptr<const Affine2DTransformation> inlineXlineToXY);

	~Grid2D();

	Grid2D& operator=(const Grid2D& other);

	/**
	 *  check all parameters
	 *
	 *  The object can still be usable even if is invalid, see specific Valid function
	 */
	bool isValid() const;
	bool isGridValid() const; // check starts, steps and count, allow negative steps
	bool isTopoValid() const; // check transformation
	bool isUnitValid() const; // check unit
	bool isSameGrid(const Grid2D& other) const;

	// make steps >= 0
	Grid2D canonized() const;

	// params
	double startInline() const;
	void setStartInline(double val);
	double startXLine() const;
	void setStartXLine(double val);
	double stepInline() const;
	void setStepInline(double val);
	double stepXLine() const;
	void setStepXLine(double val);
	long countInline() const;
	void setCountInline(long count);
	long countXLine() const;
	void setCountXLine(long count);
	SampleUnit depthAxis() const;
	void setDepthAxis(SampleUnit axis);
	std::shared_ptr<const Affine2DTransformation> inlineXlineToXY() const;
	void setInlineXlineToXY(const Affine2DTransformation& inlineXlineToXY);
	void setInlineXlineToXY(std::shared_ptr<const Affine2DTransformation> inlineXlineToXY);
	std::shared_ptr<const Affine2DTransformation> ijToXY() const;

	static double NULL_STEP_EPSILON;

	static Grid2D getMapGridFromDatasetPath(const std::string& datasetPath);

private:
	void computeIjToXY();

	double m_startInline = 0;
	double m_startXLine = 0;
	double m_stepInline = 1;
	double m_stepXLine = 1;
	long m_countInline = 0;
	long m_countXLine = 0;
	SampleUnit m_depthAxis = SampleUnit::NONE;
	std::shared_ptr<const Affine2DTransformation> m_inlineXlineToXY;
	std::shared_ptr<const Affine2DTransformation> m_ijToXY;
};

#endif // SRC_SLICER_DATA_GRID2D_H

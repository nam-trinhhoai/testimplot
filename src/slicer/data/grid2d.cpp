#include "grid2d.h"

#include "sismagedbmanager.h"
#include "smsurvey3D.h"
#include "Xt.h"

#include <gdal.h>

#include <QProcess>

#include <iostream>

double Grid2D::NULL_STEP_EPSILON = 1e-20;

Grid2D::Grid2D() {
	m_startInline = 0;
	m_startXLine = 0;
	m_stepInline = 1;
	m_stepXLine = 1;
	m_countInline = 0;
	m_countXLine = 0;
}

Grid2D::Grid2D(const Grid2D& other) {
	m_startInline = other.m_startInline;
	m_startXLine = other.m_startXLine;
	m_stepInline = other.m_stepInline;
	m_stepXLine = other.m_stepXLine;
	m_countInline = other.m_countInline;
	m_countXLine = other.m_countXLine;
	m_depthAxis = other.m_depthAxis;
	m_inlineXlineToXY = other.m_inlineXlineToXY;
	m_ijToXY = other.m_ijToXY;
}

Grid2D::Grid2D(double startInline, double startXLine, double stepInline, double stepXLine,
		long countInline, long countXLine, SampleUnit depthAxis,
		const Affine2DTransformation& inlineXlineToXY) {
	m_startInline = startInline;
	m_startXLine = startXLine;
	m_stepInline = stepInline;
	m_stepXLine = stepXLine;
	m_countInline = countInline;
	m_countXLine = countXLine;
	m_depthAxis = depthAxis;
	m_inlineXlineToXY = std::shared_ptr<const Affine2DTransformation>(new Affine2DTransformation(inlineXlineToXY));
	computeIjToXY();
}

Grid2D::Grid2D(double startInline, double startXLine, double stepInline, double stepXLine,
		long countInline, long countXLine, SampleUnit depthAxis,
		std::shared_ptr<const Affine2DTransformation> inlineXlineToXY) {
	m_startInline = startInline;
	m_startXLine = startXLine;
	m_stepInline = stepInline;
	m_stepXLine = stepXLine;
	m_countInline = countInline;
	m_countXLine = countXLine;
	m_depthAxis = depthAxis;
	m_inlineXlineToXY = inlineXlineToXY;
	computeIjToXY();
}

Grid2D::~Grid2D() {

}

Grid2D& Grid2D::operator=(const Grid2D& other) {
	m_startInline = other.m_startInline;
	m_startXLine = other.m_startXLine;
	m_stepInline = other.m_stepInline;
	m_stepXLine = other.m_stepXLine;
	m_countInline = other.m_countInline;
	m_countXLine = other.m_countXLine;
	m_depthAxis = other.m_depthAxis;
	m_inlineXlineToXY = other.m_inlineXlineToXY;
	m_ijToXY = other.m_ijToXY;
	return *this;
}

bool Grid2D::isValid() const {
	return isGridValid() && isTopoValid() && isUnitValid();
}

bool Grid2D::isGridValid() const {
	return m_countInline>0 && m_countXLine>0 && std::fabs(m_stepInline)>NULL_STEP_EPSILON && std::fabs(m_stepXLine)>NULL_STEP_EPSILON;
}

bool Grid2D::isTopoValid() const {
	return m_inlineXlineToXY!=nullptr;
}

bool Grid2D::isUnitValid() const {
	return m_depthAxis!=SampleUnit::NONE;
}

bool Grid2D::isSameGrid(const Grid2D& other) const {
	// not sure if canonized is necessary;
	Grid2D canonizedThis = canonized();
	Grid2D canonizedOther = other.canonized();

	return canonizedThis.m_countInline==canonizedOther.m_countInline && canonizedThis.m_countXLine==canonizedOther.m_countXLine &&
			std::fabs(canonizedThis.m_startInline-canonizedOther.m_startInline)<NULL_STEP_EPSILON &&
			std::fabs(canonizedThis.m_startXLine-canonizedOther.m_startXLine)<NULL_STEP_EPSILON &&
			std::fabs(canonizedThis.m_stepInline-canonizedOther.m_stepInline)<NULL_STEP_EPSILON &&
			std::fabs(canonizedThis.m_stepXLine-canonizedOther.m_stepXLine)<NULL_STEP_EPSILON;
}

Grid2D Grid2D::canonized() const {
	double newStartInline = m_startInline;
	double newStepInline = std::fabs(m_stepInline);
	long newCountInline = std::abs(m_countInline);
	bool changeInlineAxis = m_stepInline*m_countInline<0;
	if (changeInlineAxis) {
		newStartInline = m_startInline + m_stepInline * (m_countInline - 1);

	}
	double newStartXLine = m_startXLine;
	double newStepXLine = std::fabs(m_stepXLine);
	long newCountXLine = std::abs(m_countXLine);
	bool changeXLineAxis = m_stepXLine*m_countXLine<0;
	if (changeXLineAxis) {
		newStartInline = m_startXLine + m_stepXLine * (m_countXLine - 1);
	}

	return Grid2D(newStartInline, newStartXLine, newStepInline, newStepXLine, newCountInline, newCountXLine, m_depthAxis, m_inlineXlineToXY);
}

// params
double Grid2D::startInline() const {
	return m_startInline;
}

void Grid2D::setStartInline(double val) {
	m_startInline = val;
	computeIjToXY();
}

double Grid2D::startXLine() const {
	return m_startXLine;
}

void Grid2D::setStartXLine(double val) {
	m_startXLine = val;
	computeIjToXY();
}

double Grid2D::stepInline() const {
	return m_stepInline;
}

void Grid2D::setStepInline(double val) {
	m_stepInline = 0;
	computeIjToXY();
}

double Grid2D::stepXLine() const {
	return m_stepXLine;
}

void Grid2D::setStepXLine(double val) {
	m_stepXLine = val;
	computeIjToXY();
}

long Grid2D::countInline() const {
	return m_countInline;
}

void Grid2D::setCountInline(long count) {
	m_countInline = count;
	computeIjToXY();
}

long Grid2D::countXLine() const {
	return m_countXLine;
}

void Grid2D::setCountXLine(long count) {
	m_countXLine = count;
	computeIjToXY();
}

SampleUnit Grid2D::depthAxis() const {
	return m_depthAxis;
}

void Grid2D::setDepthAxis(SampleUnit axis) {
	m_depthAxis = axis;
}

std::shared_ptr<const Affine2DTransformation> Grid2D::inlineXlineToXY() const {
	return m_inlineXlineToXY;
}

void Grid2D::setInlineXlineToXY(const Affine2DTransformation& inlineXlineToXY) {
	m_inlineXlineToXY = std::shared_ptr<const Affine2DTransformation>(new Affine2DTransformation(inlineXlineToXY));
	computeIjToXY();
}

void Grid2D::setInlineXlineToXY(std::shared_ptr<const Affine2DTransformation> inlineXlineToXY) {
	m_inlineXlineToXY = inlineXlineToXY;
	computeIjToXY();
}

std::shared_ptr<const Affine2DTransformation> Grid2D::ijToXY() const {
	return m_ijToXY;
}

Grid2D Grid2D::getMapGridFromDatasetPath(const std::string& datasetPath) {
	QProcess process;
	QStringList options;
	options << QString::fromStdString(datasetPath);
	process.start("TestXtFile", options);
	process.waitForFinished();
	if (process.exitCode()!=QProcess::NormalExit) {
		std::cerr << "provided file is not in xt format (" << datasetPath << ")" << std::endl;
		return Grid2D();
	}

	inri::Xt xt(datasetPath);
	if (!xt.is_valid()) {
		return Grid2D();
	}

	double startInline = xt.startSlice();

	double startXLine = xt.startRecord();

	long countInline = xt.nSlices();

	long countXLine = xt.nRecords();

	double stepInline = xt.stepSlices();

	double stepXLine = xt.stepRecords();

	SampleUnit depthAxis = SampleUnit::NONE;
	if (xt.axis()==inri::Xt::Axis::Time) {
		depthAxis = SampleUnit::TIME;
	} else if (xt.axis()==inri::Xt::Axis::Depth) {
		depthAxis = SampleUnit::DEPTH;
	}

	// get topo
	std::string surveyPath = SismageDBManager::survey3DPathFromDatasetPath(datasetPath);
	SmSurvey3D survey3D(surveyPath);

	return Grid2D(startInline, startXLine, stepInline, stepXLine, countInline, countXLine,
			depthAxis, survey3D.inlineXlineToXYTransfo());
}

void Grid2D::computeIjToXY() {
	if (!isGridValid() || !isTopoValid()) {
		m_ijToXY = nullptr;
		return;
	}

	std::array<double, 6> inlineXlineTransfo = m_inlineXlineToXY->direct();
	std::array<double, 6> ijToInlineXline;
	ijToInlineXline[0] = m_startXLine;
	ijToInlineXline[1] = m_stepXLine;
	ijToInlineXline[2] = 0;

	ijToInlineXline[3] = m_startInline;
	ijToInlineXline[4] = 0;
	ijToInlineXline[5] = m_stepInline;

	std::array<double, 6> res;
	GDALComposeGeoTransforms(ijToInlineXline.data(), inlineXlineTransfo.data(),
					res.data());
	m_ijToXY = std::shared_ptr<const Affine2DTransformation>(new Affine2DTransformation(m_countXLine, m_countInline, res));
}

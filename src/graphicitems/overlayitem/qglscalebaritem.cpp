#include "qglscalebaritem.h"
#include <iomanip>
#include <sstream>
#include <iostream>

#include <QPainter>
#include <QOpenGLTexture>
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector2D>
#include <QPaintEngine>
#include <QStyleOptionGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>
//#include <QGLContext>
#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QMatrix2x2>

#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QDebug>

#include <iostream>
#include <cmath>

const int QGLScaleBarItem::PowerRangeMin = -5;
const int QGLScaleBarItem::PowerRangeMax = 10;
const int QGLScaleBarItem::nNiceNumber = 4;
const double QGLScaleBarItem::NiceNumberArray[] = { 1, 2, 2.5, 5 };
const double QGLScaleBarItem::metersPerInch = 0.0254;

double QGLScaleBarItem::getRoundIncrement(double startValue) {
	int nPower; //power of 10. Range of -5 to 10 gives huge scale range.
	double dMultiplier; //Mulitiplier, =10^exp, to apply to nice numbers.
	double dCandidate;  //Candidate value for new interval.

	for (nPower = PowerRangeMin; nPower <= PowerRangeMax; nPower++) {
		dMultiplier = std::pow(10.0, nPower);
		for (int i = 0; i < nNiceNumber; i++) {
			dCandidate = NiceNumberArray[i] * dMultiplier;
			if (dCandidate > startValue)
				return dCandidate;
		}
	}
	return dCandidate; //return the maximum
}

double QGLScaleBarItem::calcMapScale(double widthMap, double widthPage,
		double mapUnitFactor, int dpi) {
	double fMapWidth, fPageWidth;
	double ratio;
	if (widthPage <= 0) {
		return 0.0;
	}

	//convert map width to meters
	fMapWidth = widthMap * mapUnitFactor;

	//convert page width to meters.
	try {
		fPageWidth = widthPage / ((double) dpi) * metersPerInch;
		ratio = std::fabs(fMapWidth / fPageWidth);
	} catch (...) {
		ratio = 0.0;
	}
	return ratio;
}

void QGLScaleBarItem::calcBarScale(int width, int numTics, double fMapScale,
		double fBarUnitFactor, int &pixelsPerTic, double &unitsPerTic,
		int dpi) {
	double fBarScale;
	double fBarUnitsPerPixel;
	int nMinPixelsPerTic;

	nMinPixelsPerTic = width / (numTics - 1);

	fBarScale = fMapScale / fBarUnitFactor; //scalebar's scale.
	fBarUnitsPerPixel = fBarScale * metersPerInch / dpi;

	//calculate the result
	unitsPerTic = nMinPixelsPerTic * fBarUnitsPerPixel;
	unitsPerTic = getRoundIncrement(unitsPerTic);

	pixelsPerTic = (int) (unitsPerTic / fBarUnitsPerPixel);
}


QGLScaleBarItem::QGLScaleBarItem(const QRectF &worldExtent, QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent) {
	m_initialized = false;

	QColor c(Qt::red);
	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.5f);

	m_worldExtent = worldExtent;
	m_tickRatio = 0.1;
	m_numTicks = 5;
	m_program = new QOpenGLShaderProgram(this);

	m_mapScale=1.0f;
}

QGLScaleBarItem::~QGLScaleBarItem() {
	m_initialized = false;
}

void QGLScaleBarItem::initializeGL() {
	initShaders();

	m_nvPathFuncs.reset(new QOpenGLExtension_NV_path_rendering);
	if(!m_nvPathFuncs->initializeOpenGLFunctions())
		qDebug()<<"Failed to initialize NV PATH";

	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBuffer.allocate(4 * sizeof(float));
	m_vertexBuffer.release();

	m_ticksVertexBuffer.create();
	m_ticksVertexBuffer.bind();
	m_ticksVertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_ticksVertexBuffer.allocate(4 * m_numTicks * sizeof(float));
	m_ticksVertexBuffer.release();

	m_initialized = true;
}

void QGLScaleBarItem::initShaders() {
	if (!loadProgram(m_program, ":shaders/common/scaleBar.vert",
			":shaders/common/simpleColor.frag"))
		qDebug() << "Failed to initialize shaders";
}

void QGLScaleBarItem::setMapScale(double value)
{
	m_mapScale=value;
}


void QGLScaleBarItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	glClearStencil(0);
	glClearColor(0, 0, 0, 0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (!m_initialized)
		initializeGL();

	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}

	double mapScaleFactor = m_mapScale; //convertion from meter to someting else....
	std::string unitName = "mm";
	double scaleBarUnitInMeter = 0.001;
	if (exposed.width() > 0.1 && exposed.width() < 1) {
		scaleBarUnitInMeter = 0.01;
		unitName = "cm";
	} else if (exposed.width() > 1 && exposed.width() < 10) {
		scaleBarUnitInMeter = 0.1;
		unitName = "dm";
	} else if (exposed.width() < 10000) {
		scaleBarUnitInMeter = 1;
		unitName = "m";
	} else {
		scaleBarUnitInMeter = 1000;
		unitName = "km";
	}

	int scaleBarWidth = (int) (width * 0.2);
	double mapScale = calcMapScale(exposed.width(), width, mapScaleFactor,
			dpiX);

	int pixelPerTick;
	double unitPerTick;
	calcBarScale(scaleBarWidth, m_numTicks, mapScale, scaleBarUnitInMeter,
			pixelPerTick, unitPerTick, dpiX);
	//std::cout<<"UnitPerTick:"<<unitPerTick<<std::endl;

	int trueWidth = (m_numTicks - 1) * pixelPerTick;
	//Lines
	int borderX = 30;
	int borderY = 30;

	float pos[4];
	float w = width;
	float h = height;

	float yPos = 2 * borderY / h - 1;
	float xOrig = 2 * borderX / w - 1;
	pos[0] = xOrig;
	pos[1] = yPos;

	pos[2] = 2 * (borderX + trueWidth) / w - 1;
	pos[3] = yPos;

	QMatrix4x4 saved;
	glGetFloatv( GL_MODELVIEW_MATRIX, saved.data());

	//Draw directly into the screen
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glLineWidth(3.f);

	m_program->bind();
	m_vertexBuffer.bind();
	m_vertexBuffer.write(0, pos, 4 * sizeof(float));

	// program setup
	setupShader();
	glDrawArrays(GL_LINES, 0, 4);
	m_vertexBuffer.release();
	m_program->release();

	std::vector<float> vertex(m_numTicks * 4);
	for (int i = 0; i < m_numTicks; i++) {
		double tickSize = 0.02;
		if (i == 0 || i == m_numTicks - 1) {
			tickSize = 0.05;
			glLineWidth(2.f);
		} else
			glLineWidth(1.f);

		float offsetX = 2 * (i * pixelPerTick) / w;
		vertex[4 * i] = xOrig + offsetX;
		vertex[4 * i + 1] = yPos;

		vertex[4 * i + 2] = xOrig + offsetX;
		vertex[4 * i + 3] = yPos + tickSize;
	}
	m_program->bind();
	m_ticksVertexBuffer.bind();
	m_ticksVertexBuffer.write(0, vertex.data(), vertex.size() * sizeof(float));
	setupShader();
	glDrawArrays(GL_LINES, 0, m_numTicks * 2);
	m_ticksVertexBuffer.release();
	m_program->release();

	//Now draw the label
	const GLfloat emScale = 2048;  // match TrueType convention

	const int numChars = 256;  // ISO/IEC 8859-1 8-bit character range
//	GLuint glyphBase = glGenPathsNV(numChars + 1);
	GLuint glyphBase = m_nvPathFuncs->glGenPathsNV(numChars + 1);
	m_nvPathFuncs->glPathGlyphRangeNV(glyphBase,
	GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV, 0, numChars,
	GL_USE_MISSING_GLYPH_NV, ~0, emScale);

	GLfloat xyMinMax[4];
	m_nvPathFuncs->glGetPathMetricRangeNV(
			GL_FONT_X_MIN_BOUNDS_BIT_NV | GL_FONT_X_MAX_BOUNDS_BIT_NV
					| GL_FONT_Y_MIN_BOUNDS_BIT_NV | GL_FONT_Y_MAX_BOUNDS_BIT_NV,
			glyphBase, 1, 4 * sizeof(GLfloat), xyMinMax);

	float textSize = 8;
	double fontW = xyMinMax[1] - xyMinMax[0];
	double fontH = xyMinMax[3] - xyMinMax[2];

	double fScale = textSize * std::max(1 / fontW, 1 / fontH);
	glColor4f(m_color.x(), m_color.y(), m_color.z(), m_color.w());

	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
	glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

	std::stringstream ss;
	ss << std::fixed << std::setprecision(0) << (m_numTicks - 1) * unitPerTick
			<< " " << unitName;
	std::string text = ss.str();

	int length = text.length();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(borderX + trueWidth / 2.0, height - borderY - 15, 0);
	glScalef(fScale, -fScale, fScale);
	glTranslatef(-length * fontW / 2, 0, 0);

	drawText(glyphBase, ss.str());
	glDisable(GL_STENCIL_TEST);


	m_nvPathFuncs->glDeletePathsNV(glyphBase, numChars + 1);

	//Restore!
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(saved.data());
	glLineWidth(1.f);

}

void QGLScaleBarItem::setupShader() {
	m_program->setUniformValue("color", m_color);
	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2);
}

void QGLScaleBarItem::drawText(GLuint glyphBase, const std::string &text) {
	GLfloat xtranslate[text.length() + 1];
	xtranslate[0] = 0;
	m_nvPathFuncs->glGetPathSpacingNV(GL_ACCUM_ADJACENT_PAIRS_NV, text.length(),
	GL_UNSIGNED_BYTE, text.c_str(), glyphBase, 1.0f, 1.0f,
	GL_TRANSLATE_X_NV, &xtranslate[1]);

	m_nvPathFuncs->glStencilFillPathInstancedNV(text.length(), GL_UNSIGNED_BYTE,
			text.c_str(), glyphBase,
			GL_PATH_FILL_MODE_NV, 0xFF,
			GL_TRANSLATE_X_NV, xtranslate);

	m_nvPathFuncs->glCoverFillPathInstancedNV(text.length(), GL_UNSIGNED_BYTE,
			text.c_str(), glyphBase,
			GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
			GL_TRANSLATE_X_NV, xtranslate);
}


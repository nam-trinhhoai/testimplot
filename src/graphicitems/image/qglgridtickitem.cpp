#include "qglgridtickitem.h"
#include "igeorefimage.h"

#include <QPainter>
#include <QMatrix4x4>

#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <iostream>
#include <cmath>
#include <cstdlib>

QGLGridTickItem::QGLGridTickItem(IGeorefImage *image, QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent) {
	m_image=image;
	m_worldExtent = image->worldExtent();

	m_initialized = false;

	m_tickSize = std::min(m_worldExtent.width(), m_worldExtent.height()) * 2 / 100;
	QColor hColor=Qt::yellow;
	QColor vColor=Qt::yellow;

	m_horizontalColor = QVector4D(hColor.redF(), hColor.greenF(), hColor.blueF(), 0.85f);
	m_verticalColor = QVector4D(vColor.redF(), vColor.greenF(), vColor.blueF(), 0.85f);

	m_nvPathFuncs.reset(new QOpenGLExtension_NV_path_rendering);
	m_program = new QOpenGLShaderProgram(this);
}

QGLGridTickItem::~QGLGridTickItem() {
	m_initialized = false;
}

void QGLGridTickItem::addVerticalTick(std::vector<float> *vertexes, float pos,
		bool major) {
	int tickSize = m_tickSize;
	if (major)
		tickSize *= 2;

	float half_tick = tickSize / 2.0f;

	vertexes->push_back(half_tick);
	vertexes->push_back(pos);
	vertexes->push_back(-half_tick);
	vertexes->push_back(pos);
	if (pos > 0) {
		m_verticalTickElements.push_back(TickElement { half_tick, pos, major,
				std::to_string((int) pos) });
	}
}

void QGLGridTickItem::addHorizontalTick(std::vector<float> *vertexes, float pos,bool major) {
	int tickSize = m_tickSize;
	if (major)
		tickSize *= 2;

	float half_tick = tickSize / 2.0f;
	vertexes->push_back(pos);
	vertexes->push_back(half_tick);
	vertexes->push_back(pos);
	vertexes->push_back(-half_tick);
	if (pos > 0) {
		m_horizontalTickElements.push_back(TickElement { pos, half_tick, major,
				std::to_string((int) pos) });
	}
}

void QGLGridTickItem::initializeGL() {
	if (!loadProgram(m_program, ":shaders/common/common.vert",
			":shaders/common/simpleColor.frag"))
		qDebug() << "Failed to initialize shaders";
	m_nvPathFuncs->initializeOpenGLFunctions();

	updateInternalBuffer();
	m_initialized = true;
}

void QGLGridTickItem::updateInternalBuffer() {
	{
		std::vector<float> horVertexes;
		m_horizontalTickElements.clear();
		horVertexes.push_back(0.0f);
		horVertexes.push_back(0.0f);
		horVertexes.push_back((float) m_image->width());
		horVertexes.push_back(0.0f);

		int minorSpacing = 10;
		int majorSpacing = 100;
		if (m_image->width() > 2000) {
			minorSpacing *= 10;
			majorSpacing *= 10;
		}

		int index = 0;
		while (index < m_image->width()) {
			addHorizontalTick(&horVertexes, index, index % majorSpacing == 0);
			index += minorSpacing;
		}

		m_horizotalLineCount = horVertexes.size() / 4;

		m_horizontalVertexBuffer.create();
		m_horizontalVertexBuffer.bind();
		m_horizontalVertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
		m_horizontalVertexBuffer.allocate(horVertexes.size() * sizeof(float));
		m_horizontalVertexBuffer.write(0, horVertexes.data(), horVertexes.size() * sizeof(float));
		m_horizontalVertexBuffer.release();
	}
	{
		std::vector<float> vertVertexes;
		m_verticalTickElements.clear();
		vertVertexes.push_back(0.1f);
		vertVertexes.push_back(0.0f);
		vertVertexes.push_back(0.1f);
		vertVertexes.push_back((float) m_image->height());

		int minorSpacing = 10;
		int  majorSpacing = 100;
		if (m_image->height() > 2000) {
			minorSpacing *= 10;
			majorSpacing *= 10;
		}
		int  index = 0;
		while (index < m_image->height()) {
			addVerticalTick(&vertVertexes, index, index % majorSpacing == 0);
			index += minorSpacing;
		}

		m_verticalLineCount = vertVertexes.size() / 4;

		m_verticalVertexBuffer.create();
		m_verticalVertexBuffer.bind();
		m_verticalVertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
		m_verticalVertexBuffer.allocate(vertVertexes.size() * sizeof(float));
		m_verticalVertexBuffer.write(0, vertVertexes.data(), vertVertexes.size() * sizeof(float));
		m_verticalVertexBuffer.release();
	}
}



void QGLGridTickItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	if(!m_initialized)
		initializeGL();

	double originX=std::max(m_worldExtent.x(),exposed.x());
	double originY=std::max(m_worldExtent.y(),exposed.y());
	glClearStencil(0);
	glStencilMask(~0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glClearColor(0, 0, 0, 0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (!m_program->isLinked()) {
		if(!m_program->link())
			return;
	}
	QMatrix4x4 matrix;
	matrix.setToIdentity();
	matrix.translate(0,originY);

	// program setup
	m_horizontalVertexBuffer.bind();
	m_program->bind();
	setupShader(viewProjectionMatrix,matrix,m_horizontalColor);
	glDrawArrays(GL_LINES, 0, 2 * m_horizotalLineCount);
	m_horizontalVertexBuffer.release();
	m_program->release();

	matrix.setToIdentity();
	matrix.translate(originX,0);

	m_verticalVertexBuffer.bind();
	m_program->bind();
	setupShader(viewProjectionMatrix,matrix,m_verticalColor);
	glDrawArrays(GL_LINES, 0, 2 * m_verticalLineCount);
	m_verticalVertexBuffer.release();
	m_program->release();

	//Draw ticks text
	//Initialize glyphs
	const GLfloat emScale = 2048;  // match TrueType convention

	const int numChars = 256;  // ISO/IEC 8859-1 8-bit character range
	GLuint glyphBase = m_nvPathFuncs->glGenPathsNV(numChars + 1);
	m_nvPathFuncs->glPathParameteriNV(glyphBase, GL_PATH_STROKE_WIDTH_NV,
			2048 * 0.1);
	m_nvPathFuncs->glPathParameteriNV(glyphBase, GL_PATH_JOIN_STYLE_NV,
	GL_ROUND_NV);

	m_nvPathFuncs->glPathGlyphRangeNV(glyphBase,
	GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV, 0, numChars,
	GL_USE_MISSING_GLYPH_NV, ~0, emScale);

	//Collect YMin/Max
	GLfloat xyMinMax[4];
	m_nvPathFuncs->glGetPathMetricRangeNV(
			GL_FONT_X_MIN_BOUNDS_BIT_NV | GL_FONT_X_MAX_BOUNDS_BIT_NV
					| GL_FONT_Y_MIN_BOUNDS_BIT_NV | GL_FONT_Y_MAX_BOUNDS_BIT_NV,
			glyphBase, 1, 4 * sizeof(GLfloat), xyMinMax);

	glColor4f(m_horizontalColor.x(),m_horizontalColor.y(),m_horizontalColor.z(),m_horizontalColor.w());
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
	glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

	//Define a scale factor
	double fontW = std::fabs(xyMinMax[1] - xyMinMax[0]);
	double fontH = std::fabs(xyMinMax[2] - xyMinMax[3]);

	float fontPixelSize=6.0f;

	double fScaleX = exposed.width() / width *fontPixelSize / fontW;
	double fScaleY = exposed.height() / height *fontPixelSize / fontH;


	QMatrix4x4 saved;
	glGetFloatv( GL_MODELVIEW_MATRIX, saved.data());

	double offsetX = fScaleX * (xyMinMax[1] - xyMinMax[0]);
	double offsetY = fScaleY * (xyMinMax[3] - xyMinMax[2]);

	for (TickElement t : m_horizontalTickElements) {
		if (!t.major)
			continue;
		int length=t.text.length();
		QMatrix4x4 test;
		test.setToIdentity();
		test.translate(t.x-length *offsetX/2, originY+t.y-2*offsetY);
		test.scale(fScaleX, -fScaleY, 1);
		test = saved * test;
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(test.data());
		drawText(glyphBase, t.text);
	}
	for (TickElement t : m_verticalTickElements) {
			if (!t.major)
				continue;
		int length=t.text.length();
		QMatrix4x4 test;
		test.setToIdentity();
		test.translate(originX+t.x+offsetX, t.y-offsetY/2);
		test.scale(fScaleX, -fScaleY, 1);
		test = saved * test;
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(test.data());
		drawText(glyphBase, t.text);
	}

	glDisable(GL_STENCIL_TEST);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(saved.data());

	m_nvPathFuncs->glDeletePathsNV(glyphBase, numChars + 1);
}

void QGLGridTickItem::setupShader(const QMatrix4x4 &viewProjectionMatrix,const QMatrix4x4 &matrix, const QVector4D & color)
{
	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", color);

	m_program->setUniformValue("transfoMatrix", matrix);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2);
}

void QGLGridTickItem::drawText(GLuint glyphBase, const std::string &text) {
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




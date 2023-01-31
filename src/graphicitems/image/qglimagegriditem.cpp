#include "qglimagegriditem.h"
#define _USE_MATH_DEFINES
#include <cmath>
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
#include "igeorefimage.h"

#include <iostream>

QGLImageGridItem::QGLImageGridItem(
		const IGeorefImage * const provider,
		QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent),m_provider(provider){
	m_initialized = false;
	m_transfoMatrix = provider->imageToWorldTransformation();
	int width=provider->width();
	int height=provider->height();

	double p0x, p0y;
	double p1x, p1y;
	double p2x, p2y;

	provider->imageToWorld(0, 0, p0x, p0y);
	provider->imageToWorld(width, 0, p1x, p1y);
	provider->imageToWorld(0, height, p2x, p2y);

	if (std::abs(p1y - p0y) > 0.000000001 && std::abs(p1x - p0x) > 0.000000001) {
		angleTextX = std::atan2((p1x - p0x), (p1y - p0y)) * 180 / M_PI;
		angleTextY = std::atan2((p2x - p0x), (p2y - p0y)) * 180 / M_PI;
	} else  {
		angleTextX = 90;
		angleTextY = 0;
	}
	QColor c(Qt::cyan);
	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.5f);

	m_worldExtent = IGeorefImage::worldExtent(provider); //in theory we should plan to add a little offset because of the grids info

	m_width = width;
	m_height = height;

	m_tickInterval = 400;

	m_needUpdate = true;

	m_program = new QOpenGLShaderProgram(this);
}

void QGLImageGridItem::setColor(QColor c) {
	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.5f);
	update();
}

QGLImageGridItem::~QGLImageGridItem() {
	m_initialized = false;
}

void QGLImageGridItem::initializeGL() {
	initShaders();

	m_nvPathFuncs.reset(new QOpenGLExtension_NV_path_rendering);
	m_nvPathFuncs->initializeOpenGLFunctions();
	m_initialized = true;
}

void QGLImageGridItem::updateInternalBuffer() {
	if (!m_needUpdate)
		return;

	std::vector<float> vertices;

	vertices.push_back(0);
	vertices.push_back(0);
	vertices.push_back(m_width);
	vertices.push_back(0);

	vertices.push_back(0);
	vertices.push_back(m_height);
	vertices.push_back(m_width);
	vertices.push_back(m_height);


	vertices.push_back(0);
	vertices.push_back(0);
	vertices.push_back(0);
	vertices.push_back(m_height);

	vertices.push_back(m_width);
	vertices.push_back(0);
	vertices.push_back(m_width);
	vertices.push_back(m_height);

	m_lineCount = vertices.size() / 2;
	if (m_vertexBuffer.isCreated())
		m_vertexBuffer.destroy();

	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_vertexBuffer.allocate(vertices.size() * sizeof(float));
	m_vertexBuffer.write(0, vertices.data(), vertices.size() * sizeof(float));
	m_vertexBuffer.release();
	m_needUpdate = false;
}

void QGLImageGridItem::initShaders() {
	if(!loadProgram(m_program,":shaders/common/common.vert",":shaders/common/simpleColor.frag"))
		qDebug() << "Failed to initialize shaders";
}

void QGLImageGridItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX,int dpiY) {


	updateInternalBuffer();

	glClearStencil(0);
	glClearColor(0, 0, 0, 0);
	glClear(GL_STENCIL_BUFFER_BIT);

	if (!m_initialized)
		initializeGL();

	if (!m_program->isLinked()) {
		m_program->link();
	}
	if (!m_program->isLinked()) {
		qDebug() << "Shader not linked";
		return;
	}

	// program setup
	m_program->bind();
	m_vertexBuffer.bind();
	setupShader(viewProjectionMatrix, m_transfoMatrix);
	glDrawArrays(GL_LINES, 0, 2 * m_lineCount);
	m_vertexBuffer.release();
	m_program->release();

	glColor3f(m_color.x(),m_color.y(),m_color.z());

	//	//Draw ticks text
	const GLfloat emScale = 2048;  // match TrueType convention

	const int numChars = 256;  // ISO/IEC 8859-1 8-bit character range
	GLuint glyphBase = m_nvPathFuncs->glGenPathsNV(numChars + 1);
	m_nvPathFuncs->glPathGlyphRangeNV(glyphBase,
			GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV, 0, numChars,
			GL_USE_MISSING_GLYPH_NV, ~0, emScale);

	GLfloat xyMinMax[4];
	m_nvPathFuncs->glGetPathMetricRangeNV(
			GL_FONT_X_MIN_BOUNDS_BIT_NV | GL_FONT_X_MAX_BOUNDS_BIT_NV
			| GL_FONT_Y_MIN_BOUNDS_BIT_NV | GL_FONT_Y_MAX_BOUNDS_BIT_NV,
			glyphBase, 1, 4 * sizeof(GLfloat), xyMinMax);

	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
	glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

	//Define a scale factor
	double fontW=xyMinMax[1]-xyMinMax[0];
	double fontH=xyMinMax[3]-xyMinMax[2];

	double fontScale=std::max(1/fontW,1/fontH);

	//Compute the current view scale
	float fScale = exposed.width() / width * 10.0 *fontScale;

	QMatrix4x4 saved;
	glGetFloatv( GL_MODELVIEW_MATRIX, saved.data());

	glDisable(GL_STENCIL_TEST);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(saved.data());
	m_nvPathFuncs->glDeletePathsNV(glyphBase, numChars + 1);
}

void QGLImageGridItem::setupShader(const QMatrix4x4 &viewProjectionMatrix,
		const QMatrix4x4 &transfo) {
	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", m_color);
	m_program->setUniformValue("transfoMatrix", transfo);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2);
}

void QGLImageGridItem::drawText(GLuint glyphBase, const std::string &text) {
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


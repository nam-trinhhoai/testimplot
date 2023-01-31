#include "qglgridaxisitem.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <QPainter>
#include <QMatrix4x4>

#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>



QGLGridAxisItem::QGLGridAxisItem(const QRectF &worldExtent,
		int expectedPixelsDim, Direction dir, QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent) {
	m_dir = dir;

	QRectF extent = worldExtent;
	if (m_dir == Direction::HORIZONTAL)
		extent.setHeight(expectedPixelsDim);
	else
		extent.setWidth(expectedPixelsDim);

	m_worldExtent = extent;
	m_initialized = false;
	QColor hColor = Qt::white;
	m_color = QVector4D(hColor.redF(), hColor.greenF(), hColor.blueF(), 0.65f);

	m_tickRatio = 0.1;
	m_transfoMatrix.setToIdentity();

	m_nvPathFuncs.reset(new QOpenGLExtension_NV_path_rendering);
	m_program = new QOpenGLShaderProgram(this);
}

QGLGridAxisItem::~QGLGridAxisItem() {
	m_initialized = false;
}


void QGLGridAxisItem::initializeOpenGLFunctions()
{
	//QOpenGLContext *context = QOpenGLContext::currentContext();
	//m_getPathSpacingNV = reinterpret_cast<void (QOPENGLF_APIENTRYP)(GLenum , GLsizei , GLenum , const GLvoid *, GLuint , GLfloat , GLfloat , GLenum , GLfloat *)>(context->getProcAddress("glGetPathSpacingNV"));

}

void QGLGridAxisItem::initializeGL() {

	if (!loadProgram(m_program, ":shaders/common/common.vert",
			":shaders/common/simpleColor.frag"))
		qDebug() << "Failed to initialize shaders";

	m_nvPathFuncs->initializeOpenGLFunctions();


	m_initialized = true;

	if (m_dir == Direction::HORIZONTAL) {
		m_numMaxPoints*=2;
		float intervalW = m_tickRatio * m_worldExtent.width();
		m_numMaxPoints = 2 * (m_worldExtent.width() / intervalW + 1);
	} else {
		m_numMaxPoints*=3;
		float intervalH = m_tickRatio * m_worldExtent.height();
		m_numMaxPoints = 2 * (m_worldExtent.height() / intervalH + 1);
	}

	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBuffer.allocate(2 * m_numMaxPoints * sizeof(float));
	m_vertexBuffer.release();
	m_initialized = true;
}

void QGLGridAxisItem::UpdateVertex(const QRectF &exposed){
	float interval;
	if (m_dir == Direction::HORIZONTAL) {
		interval = round(m_tickRatio * exposed.width()/10)*10;
		m_numMaxPoints = 2 * (exposed.width() / interval + 1);
	} else {
		interval = round(m_tickRatio * exposed.height()/10)*10;
		m_numMaxPoints = 2 * (exposed.height() / interval + 1);
	}

	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBuffer.allocate(2 * m_numMaxPoints * sizeof(float));
	m_vertexBuffer.release();
}

int QGLGridAxisItem::AddTicksDefault(const QRectF &exposed, int width,	int height)
{
	bool vertical = true;
	int lineCount;
	float pix,rangeMin,rangeMax,rangeSize;
	if (m_dir == Direction::HORIZONTAL) {
		vertical = false;
		pix  = width;
		rangeMin = exposed.x();
		rangeMax = exposed.width() + rangeMin;
		rangeSize = rangeMax - rangeMin;
	}else{
		pix  = height;
		rangeMin = exposed.y();
		rangeMax = exposed.height() + rangeMin;
		rangeSize = rangeMax - rangeMin;
	}
	const int idx0          = 0;
	const int nMinor        = 10;
	const int nMajor        = nvMax(2, (int)NV_ROUND(pix / (vertical ? 300.0f : 400.0f)));
	const double nice_range = NiceNum(rangeSize* 0.99, false);
	const double interval   = NiceNum(nice_range / (nMajor - 1), true);
	const double graphmin   = floor(rangeMin / interval) * interval;
	const double graphmax   = ceil(rangeMax/ interval) * interval;
	bool first_major_set    = false;
	int  first_major_idx    = 0;
	std::vector<float> vertices;

	std::vector<double> tickElements;
	m_tickElements.clear();

	int w = ceil(m_worldExtent.width());
	float y = ceil(exposed.y());
	float h = ceil(exposed.height());

	if (m_dir == Direction::HORIZONTAL) {
		vertices.push_back(graphmin);
		vertices.push_back(1);

		vertices.push_back(graphmax + 0.5 * interval);
		vertices.push_back(1);
	}else {
		vertices.push_back(w);
		vertices.push_back(y);

		vertices.push_back(w);
		vertices.push_back(y + h);
	}

	for (double major = graphmin; major < graphmax + 0.5 * interval; major += interval)
	{
		if (major - interval < 0 && major + interval > 0)
		{
			major = 0;
		}

		if (major >= rangeMin && major <= rangeMax)
		{
			if (!first_major_set)
			{
				first_major_set = true;
			}

			if (m_dir == Direction::HORIZONTAL) {
				vertices.push_back(major);
				vertices.push_back(5);

				vertices.push_back(major);
				vertices.push_back(1);
			}else{
				vertices.push_back(w);
				vertices.push_back(major);

				vertices.push_back(w - 5);
				vertices.push_back(major);
			}
			m_tickElements.push_back(major);
		}

		for (int i = 1; i < nMinor; ++i)
		{
			double minor = major + i * interval / nMinor;

			if (minor >= rangeMin && minor <= rangeMax)
			{
				m_tickElements.push_back(minor);

				if (m_dir == Direction::HORIZONTAL) {
					vertices.push_back(minor);
					vertices.push_back(5);

					vertices.push_back(minor);
					vertices.push_back(1);
				}else{
					vertices.push_back(w);
					vertices.push_back(minor);

					vertices.push_back(w - 5);
					vertices.push_back(minor);
				}
			}
		}
	}

	lineCount = std::min(m_numMaxPoints, (int) (vertices.size() / 2));
	m_vertexBuffer.write(0, vertices.data(), 2 * lineCount * sizeof(float));

	return lineCount;

}

int QGLGridAxisItem::updateInternalBuffer(const QRectF &exposed, int width,	int height) {
	int lineCount;

	lineCount = AddTicksDefault(exposed,width,height);

	return lineCount;
}

void QGLGridAxisItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	glClearStencil(0);
	glStencilMask(~0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glClearColor(0, 0, 0, 0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (!m_initialized)
		initializeGL();

	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}

	//Draw ticks text
	//Initialize glyphs
	const GLfloat emScale = 2048;  // match TrueType convention
	const int numChars = 256;  // ISO/IEC 8859-1 8-bit character range

	GLuint glyphBase = m_nvPathFuncs->glGenPathsNV(numChars + 1);

	m_nvPathFuncs->glPathGlyphRangeNV(glyphBase,GL_STANDARD_FONT_NAME_NV, "Sans", 0x1, 0, numChars,	GL_USE_MISSING_GLYPH_NV, ~0, emScale);

	//Collect YMin/Max
	GLfloat xyMinMax[4];
	m_nvPathFuncs->glGetPathMetricRangeNV(GL_FONT_X_MIN_BOUNDS_BIT_NV | GL_FONT_X_MAX_BOUNDS_BIT_NV	| GL_FONT_Y_MIN_BOUNDS_BIT_NV | GL_FONT_Y_MAX_BOUNDS_BIT_NV,glyphBase, 1, 4 * sizeof(GLfloat), xyMinMax);

	//Define a scale factor
	double fontW = std::fabs(xyMinMax[1] - xyMinMax[0]);
	double fontH = std::fabs(xyMinMax[2] - xyMinMax[3]);
	float fontPixelSize = 7.0f;
	double fScaleX = exposed.width() / width * fontPixelSize / fontW;
	double fScaleY = exposed.height() / height * fontPixelSize / fontH;
	QMatrix4x4 saved;
	glGetFloatv( GL_MODELVIEW_MATRIX, saved.data());
	double offsetX = fScaleX * (xyMinMax[1] - xyMinMax[0]);
	double offsetY = fScaleY * (xyMinMax[3] - xyMinMax[2]);

	// program setup
	m_program->bind();
	m_vertexBuffer.bind();
	int lineCount = updateInternalBuffer(exposed, width, height);
	setupShader(viewProjectionMatrix, m_transfoMatrix, m_color);
	glDrawArrays(GL_LINES, 0, lineCount);
	m_vertexBuffer.release();
	m_program->release();

	glColor4f(m_color.x(), m_color.y(), m_color.z(), m_color.w());
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
	glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

	int interval = 1;

	if(m_tickElements.size() > lineCount){
		interval = 2;
	}

	for (int  i = 0 ;i< m_tickElements.size();i+=interval){

		double pos = m_tickElements[i];

		std::stringstream ss;
		std::string text;
		int length ;

		if(std::abs(pos) > 9999999){
			ss << std::scientific << std::setprecision(4) << pos;
		}else if(std::abs(pos) > 999999){
			ss << std::scientific << std::setprecision(3) << pos;
		}else if(std::abs(pos) > 99999){
			ss << std::scientific << std::setprecision(2) << pos;
		} else if(std::abs(pos) > 9999){
			ss << std::scientific << std::setprecision(1) << pos;
		}else {
			if(std::abs(pos) < 1000) {
				ss << std::fixed << std::setprecision(2) << pos;
			}else{
				ss << std::fixed << std::setprecision(0) << pos;
			}
		}
		text = ss.str();
		length = text.length();

		QMatrix4x4 test;
		test.setToIdentity();
		if (m_dir == Direction::HORIZONTAL) {
			test.translate(pos- length * offsetX / 2, -2 * offsetY);
		}else{
			test.translate(m_worldExtent.width() - (length + 1) * offsetX - 5,pos + offsetY / 2);
		}
		test.scale(fScaleX, fScaleY, 1);
		test = saved * test;
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(test.data());
		drawText(glyphBase, ss.str());
	}
	glDisable(GL_STENCIL_TEST);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(saved.data());

	m_nvPathFuncs->glDeletePathsNV(glyphBase, numChars + 1);
}

void QGLGridAxisItem::setupShader(const QMatrix4x4 &viewProjectionMatrix,
		const QMatrix4x4 &matrix, const QVector4D &color) {
	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", color);

	m_program->setUniformValue("transfoMatrix", matrix);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2);
}

void QGLGridAxisItem::drawText(GLuint glyphBase, const std::string &text) {
	GLfloat xtranslate[text.length() + 1];
	xtranslate[0] = 0;
/*	m_getPathSpacingNV(GL_ACCUM_ADJACENT_PAIRS_NV, text.length(),
			GL_UNSIGNED_BYTE, text.c_str(), glyphBase, 1.0f, 1.0f,
			GL_TRANSLATE_X_NV, &xtranslate[1]);*/
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


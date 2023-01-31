#include "qglfixedaxisitem.h"
#include "igeorefimage.h"
#include "affinetransformation.h"

#include <QPainter>
#include <QMatrix4x4>

#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <iostream>
#include <cmath>
#include <cstdlib>

QGLFixedAxisItem::QGLFixedAxisItem(IGeorefImage *image,int expectedPixelsDim, int tickSize, Direction dir,QGraphicsItem *parent) :QAbstractGLGraphicsItem(parent) {
	m_dir=dir;
	m_textFlip = false;

	m_image = image;
	QRectF extent = image->worldExtent();
	if(m_dir==Direction::HORIZONTAL)
	{
		extent.setY(0);
		extent.setHeight(expectedPixelsDim);
	}else
	{
		extent.setX(0);
		extent.setWidth(expectedPixelsDim);
	}
	m_worldExtent = extent;
	m_initialized = false;
	m_tickSize = tickSize;
	QColor hColor = Qt::white;
	m_color = QVector4D(hColor.redF(), hColor.greenF(), hColor.blueF(), 0.65f);

	m_nvPathFuncs.reset(new QOpenGLExtension_NV_path_rendering);
	m_program = new QOpenGLShaderProgram(this);

	m_displayValueTransform = new AffineTransformation(1, 0, this);
}

QGLFixedAxisItem::~QGLFixedAxisItem() {
	m_initialized = false;
}

void QGLFixedAxisItem::addTick(std::vector<float> &rVertexes, QGLFixedAxisItem::Direction dir,float pos,int index,float maxLen,const QRectF exposed) {
	int tickSize = m_tickSize;
	int minorSpacing = 10;
	int majorSpacing = 100;
	bool major = false;
	int h=exposed.height()-1;
	int w=exposed.width()-1;

	if ((exposed.width() * m_displayValueTransform->a() > 2000 && m_dir == Direction::HORIZONTAL) ||
			(exposed.height() * m_displayValueTransform->a() > 2000 && m_dir == Direction::VERTICAL)){
		minorSpacing *= 10;
		majorSpacing *= 10;
	}

	while (pos < maxLen)
	{
		if((index % majorSpacing) == 0){
			tickSize*=2;
			major = true;
		}else{
			tickSize=m_tickSize;
			major = false;
		}

		double convertedPos;
		m_displayValueTransform->indirect(pos, convertedPos);

		switch(dir)
		{
		case Direction::HORIZONTAL:
		{
			rVertexes.push_back(convertedPos);
			rVertexes.push_back(h-tickSize);
			rVertexes.push_back(convertedPos);
			rVertexes.push_back(h);

			if(pos>0){
				m_tickElements.push_back(TickElement { convertedPos, tickSize, major,std::to_string((int) pos) });
			}
		}
		break;

		case Direction::VERTICAL:
		{
			rVertexes.push_back(w-tickSize);
			rVertexes.push_back(convertedPos);
			rVertexes.push_back(w);
			rVertexes.push_back(convertedPos);

			if (pos> 0){
				m_tickElements.push_back(TickElement { tickSize, convertedPos, major,std::to_string((int) pos) });
			}
		}
		break;

		default:
			break;
		}

		index += minorSpacing;
		pos += minorSpacing;
	}
}

void QGLFixedAxisItem::addHorizontalTick(std::vector<float> &rVertexes, float pos,bool major) {
	int tickSize = m_tickSize;
	if (major)
		tickSize *= 2;

	int h = m_worldExtent.height()-1;

	double convertedPos;
	m_displayValueTransform->indirect(pos, convertedPos);

	rVertexes.push_back(convertedPos);
	rVertexes.push_back(h-tickSize);
	rVertexes.push_back(convertedPos);
	rVertexes.push_back(h);

	if(pos>0){
		m_tickElements.push_back(TickElement { convertedPos, tickSize, major,std::to_string((int) pos) });
	}
}

void QGLFixedAxisItem::initializeGL() {
	if (!loadProgram(m_program, ":shaders/common/common.vert",":shaders/common/simpleColor.frag"))
		qDebug() << "Failed to initialize shaders";
	m_nvPathFuncs->initializeOpenGLFunctions();

	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);

	m_lineCount = 500;
	m_vertexBuffer.allocate(m_lineCount * sizeof(float));
	m_tickElements.clear();

	m_initialized = true;
}

void QGLFixedAxisItem::addLine(std::vector<float> &rVertexes,QGLFixedAxisItem::Direction dir,const QRectF exposed)
{
	switch(dir)
	{
	case Direction::HORIZONTAL:
	{
		rVertexes.push_back(exposed.x());
		rVertexes.push_back(exposed.height()-1);
		rVertexes.push_back(exposed.x() + exposed.width());
		rVertexes.push_back(exposed.height()-1);
	}
	break;
	case Direction::VERTICAL:
	{
		rVertexes.push_back(exposed.width()-1);
		rVertexes.push_back(exposed.y());
		rVertexes.push_back(exposed.width()-1);
		rVertexes.push_back(exposed.y() + exposed.height());
	}
	break;
	default:
		break;
	}
}

void QGLFixedAxisItem::updateInternalBuffer(const QRectF &exposed, int width, int height)//()
{
	double rangeMin,rangeMax,rangeSize;
	std::vector<float> vertexes;
	int index,nMajor;
	double pos,interval,nice_range;
	double maxLen;
	int minorSpacing = 10;
	int majorSpacing = 100;

	if ((exposed.width() * m_displayValueTransform->a() > 2000 && m_dir == Direction::HORIZONTAL) ||
			(exposed.height() * m_displayValueTransform->a() > 2000 && m_dir == Direction::VERTICAL)) {
		minorSpacing *= 10;
		majorSpacing *= 10;
	}

	m_tickElements.clear();

	if (m_dir == Direction::HORIZONTAL)
	{
		m_displayValueTransform->direct(exposed.x(), rangeMin);
		m_displayValueTransform->direct(exposed.x()+exposed.width(), rangeMax);
		rangeSize = rangeMax - rangeMin;

		nMajor = nvMax(2, (int)NV_ROUND( exposed.width() * m_displayValueTransform->a() / 400.0f));
		nice_range = NiceNum(rangeSize* 0.99, false);
		interval = NiceNum(nice_range / (nMajor - 1), true);
		pos = floor(rangeMin / interval) * interval;

		addLine(vertexes,m_dir,exposed);
		addHorizontalTick(vertexes, pos, true);

		index = minorSpacing;
		pos += minorSpacing;
		m_displayValueTransform->direct(exposed.x() + exposed.width(), maxLen);
	}
	else
	{
		m_displayValueTransform->direct(exposed.y(), rangeMin);
		m_displayValueTransform->direct(exposed.y()+exposed.height(), rangeMax);
		rangeSize = rangeMax - rangeMin;

		nMajor  = nvMax(2, (int)NV_ROUND( exposed.height() * m_displayValueTransform->a() / 300.0f));
		nice_range = NiceNum(rangeSize* 0.99, false);
		interval   = NiceNum(nice_range / (nMajor - 1), true);
		pos = floor(rangeMin / interval) * interval;

		addLine(vertexes,m_dir,exposed);
		m_displayValueTransform->direct(exposed.y() + exposed.height(), maxLen);
		index = 0;
	}

	addTick(vertexes,m_dir,pos,index,maxLen,exposed);

	m_lineCount = vertexes.size() / 4;
	m_vertexBuffer.allocate(vertexes.size() * sizeof(float));
	m_vertexBuffer.write(0, vertexes.data(),vertexes.size() * sizeof(float));
}

void QGLFixedAxisItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	if (!m_initialized)
		initializeGL();

	glClearStencil(0);
	glStencilMask(~0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glClearColor(0, 0, 0, 0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}
	QMatrix4x4 matrix;
	matrix.setToIdentity();

	// program setup
	m_vertexBuffer.bind();
	m_program->bind();
	setupShader(viewProjectionMatrix, matrix, m_color);
	QRectF newBounds = exposed.united(m_worldExtent);
	updateInternalBuffer(newBounds,width,height);
	glDrawArrays(GL_LINES, 0, 2 * m_lineCount);
	m_vertexBuffer.release();
	m_program->release();

	//Draw ticks text
	//Initialize glyphs
	const GLfloat emScale = 2048;  // match TrueType convention

	const int numChars = 256;  // ISO/IEC 8859-1 8-bit character range
	GLuint glyphBase = m_nvPathFuncs->glGenPathsNV(numChars + 1);
	m_nvPathFuncs->glPathGlyphRangeNV(glyphBase,
			GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV, 0, numChars,
			GL_USE_MISSING_GLYPH_NV, ~0, emScale);

	//Collect YMin/Max
	GLfloat xyMinMax[4];
	m_nvPathFuncs->glGetPathMetricRangeNV(
			GL_FONT_X_MIN_BOUNDS_BIT_NV | GL_FONT_X_MAX_BOUNDS_BIT_NV
			| GL_FONT_Y_MIN_BOUNDS_BIT_NV | GL_FONT_Y_MAX_BOUNDS_BIT_NV,
			glyphBase, 1, 4 * sizeof(GLfloat), xyMinMax);

	glColor4f(m_color.x(), m_color.y(), m_color.z(), m_color.w());
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
	glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

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

	for (TickElement t : m_tickElements) {
		if (!t.major)
			continue;
		int length = t.text.length();
		QMatrix4x4 test;
		test.setToIdentity();
		int coef = 1;
		if (m_dir == Direction::HORIZONTAL) {
			test.translate(t.x - length * offsetX / 2,- 2 * offsetY);
			coef = -1;
		}else{
			test.translate( m_worldExtent.width()-(length+1) * offsetX-t.x , t.y - offsetY / 2);
		}
		if (m_textFlip) {
			test.scale(coef*fScaleX, coef*fScaleY, 1);
		} else {
			test.scale(fScaleX, -fScaleY, 1);
		}
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

void QGLFixedAxisItem::setupShader(const QMatrix4x4 &viewProjectionMatrix,
		const QMatrix4x4 &matrix, const QVector4D &color) {
	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", color);

	m_program->setUniformValue("transfoMatrix", matrix);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2);
}

void QGLFixedAxisItem::drawText(GLuint glyphBase, const std::string &text) {
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

void QGLFixedAxisItem::setTextFlip(bool toggle) {
	if (toggle!=m_textFlip) {
		m_textFlip = toggle;
		update();
	}
}

bool QGLFixedAxisItem::textFlip() const {
	return m_textFlip;
}

void QGLFixedAxisItem::setDisplayValueTransform(const AffineTransformation* affineTransform) {
	m_displayValueTransform->deleteLater();

	m_displayValueTransform = new AffineTransformation(*affineTransform);
	m_displayValueTransform->setParent(this);
}

const AffineTransformation* QGLFixedAxisItem::displayValueTransform() const {
	return m_displayValueTransform;
}

QColor QGLFixedAxisItem::color() {
	QColor colorObj= QColor::fromRgbF(m_color.x(), m_color.y(), m_color.z());
	return colorObj;
}

void QGLFixedAxisItem::setColor(const QColor& newColor) {
	m_color = QVector4D(newColor.redF(), newColor.greenF(), newColor.blueF(), 0.65f);
}

void QGLFixedAxisItem::resetColor() {
	QColor hColor = Qt::white;
	m_color = QVector4D(hColor.redF(), hColor.greenF(), hColor.blueF(), 0.65f);
}

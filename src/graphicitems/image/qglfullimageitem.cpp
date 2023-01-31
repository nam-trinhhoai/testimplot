#include "qglfullimageitem.h"

#include <iostream>
#include <iomanip>
#include <limits>

#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include "qglabstractfullimage.h"
#include "qvertex2D.h"
#include "qglimageitemhelper.h"

#define COLOR_MAP_TEXTURE_UNIT 1
#define IMAGE_TEXTURE_UNIT 0

QGLFullImageItem::QGLFullImageItem(QGLAbstractFullImage *image,
		QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent) {
	m_image = image;

	//m_worldExtent is only needed to define the BBOX of the item for organizing the view
	//and allow a zoom reset. No link with "true" position. Position is hanlded by the OpenGL geometry
	m_worldExtent = image->worldExtent();
	m_initialized = false;

	m_program = new QOpenGLShaderProgram(this);
}

QGLFullImageItem::~QGLFullImageItem() {

}

void QGLFullImageItem::preInitGL()
{
}
void QGLFullImageItem::postInitGL()
{

}
void QGLFullImageItem::initializeShaders() {
	m_program->bind();
	bool done = false;
	do {
		ImageFormats::QColorFormat colorFormat = m_image->colorFormat();
		if (colorFormat == ImageFormats::QColorFormat::RGB_INTERLEAVED
				|| colorFormat
						== ImageFormats::QColorFormat::RGBA_INTERLEAVED) {
			if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",
					":shaders/images/lut_ifrag.glsl"))
				break;

		} else {
			ImageFormats::QSampleType type = m_image->sampleType();
			if (type == ImageFormats::QSampleType::UINT8
					|| type == ImageFormats::QSampleType::UINT16
					|| type == ImageFormats::QSampleType::UINT32) {
				if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",
						":shaders/images/lut_ufrag.glsl"))
					break;
			} else if (type == ImageFormats::QSampleType::INT8
					|| type == ImageFormats::QSampleType::INT16
					|| type == ImageFormats::QSampleType::INT32) {
				if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",
						":shaders/images/lut_ifrag.glsl"))
					break;
			} else {
				if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",
						":shaders/images/lut_frag.glsl"))
					break;
			}
		}
		done = true;
	} while (0);
	if (!done)
		qDebug() << "Ooops! 1";

	m_program->release();
}

void QGLFullImageItem::initializeGL() {
	preInitGL();
	initializeShaders();

	// vertex buffer initialisation
	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	// room for 2 triangles of 3 vertices
	m_vertexBuffer.allocate(2 * 3 * sizeof(QVertex2D));

	QVertex2D v0, v1, v2, v3;
	QGLImageItemHelper::computeImageCorner(m_image, v0, v1, v2, v3);

	int vCount = 0;
	// first triangle v0, v1, v2
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &v0, sizeof(QVertex2D));
	vCount++;
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &v1, sizeof(QVertex2D));
	vCount++;
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &v2, sizeof(QVertex2D));
	vCount++;

	// second triangle v1, v3, v2
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &v1, sizeof(QVertex2D));
	vCount++;
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &v3, sizeof(QVertex2D));
	vCount++;
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &v2, sizeof(QVertex2D));
	vCount++;

	m_vertexBuffer.release();

	m_image->bindLUTTexture(COLOR_MAP_TEXTURE_UNIT);
	m_image->releaseLUTTexture(COLOR_MAP_TEXTURE_UNIT);

	m_image->bindTexture(IMAGE_TEXTURE_UNIT);
	m_image->releaseTexture(IMAGE_TEXTURE_UNIT);

	postInitGL();
	m_initialized = true;
}

void QGLFullImageItem::setPaletteParameter(QOpenGLShaderProgram *program) {
	QVector2D r = m_image->rangeRatio();
	program->setUniformValue("f_rangeMin", r[0]);
	program->setUniformValue("f_rangeRatio", r[1]);
	program->setUniformValue("f_noHasDataValue", m_image->hasNoDataValue());
	program->setUniformValue("f_noDataValue", m_image->noDataValue());
}

void QGLFullImageItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (!m_initialized)
		initializeGL();


	// program setup
	m_program->bind();
	// a good practice if you have to manage multiple shared context
	// with shared resources: the link is context dependant.
	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}

	// binding the buffer
	m_vertexBuffer.bind();

	// setup of the program attributes
	int pos = 0, count;
	// positions : 2 floats
	count = 2;
	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, pos, count,
			sizeof(QVertex2D));
	pos += count * sizeof(float);

	// texture coordinates : 2 floats
	count = 2;
	m_program->enableAttributeArray("textureCoordinates");
	m_program->setAttributeBuffer("textureCoordinates", GL_FLOAT, pos, count,
			sizeof(QVertex2D));
	pos += count * sizeof(float);

	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("f_opacity", m_image->opacity());
	setPaletteParameter(m_program);

	m_program->setUniformValue("color_map", COLOR_MAP_TEXTURE_UNIT);
	m_program->setUniformValue("f_tileTexture", IMAGE_TEXTURE_UNIT);

	m_image->bindLUTTexture(COLOR_MAP_TEXTURE_UNIT);
	m_image->bindTexture(IMAGE_TEXTURE_UNIT);
	// draw 2 triangles = 6 vertices starting at offset 0 in the buffer
	glDrawArrays(GL_TRIANGLES, 0, 6);
	// release texture
	m_image->releaseTexture(IMAGE_TEXTURE_UNIT);
	m_image->releaseLUTTexture(COLOR_MAP_TEXTURE_UNIT);

	m_vertexBuffer.release();
	m_program->release();
}


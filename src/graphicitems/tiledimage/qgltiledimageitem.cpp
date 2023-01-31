#include "qgltiledimageitem.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include "qglabstracttiledimage.h"
#include "qglrenderthread.h"

#define COLOR_MAP_TEXTURE_UNIT 1
#define IMAGE_TEXTURE_UNIT 0

QGLTiledImageItem::QGLTiledImageItem(QGLAbstractTiledImage *image, QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent) {
	m_transparentTile = nullptr;

	m_thread = new QGLRenderThread(image, this);
	m_thread->startService();
	connect(image, SIGNAL(cachedImageInserted(QGLTile *)),
			SLOT(cacheImageInserted(QGLTile *)));

	m_image = image;

	//m_worldExtent is only needed to define the BBOX of the item for organizing the view
	//and allow a zoom reset. No link with "true" position. Position is hanlded by the OpenGL geometry
	m_worldExtent = m_image->worldExtent();
	m_initialized = false;

	m_program = new QOpenGLShaderProgram(this);
}
void QGLTiledImageItem::cacheImageInserted(QGLTile *tile) {
	//Invalidate the area
	update(tile->coords().worldBoundingRect());
}
QGLTiledImageItem::~QGLTiledImageItem() {

	if (m_thread)
		m_thread->stopService();
}

void QGLTiledImageItem::initializeShaders() {
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

void QGLTiledImageItem::setPaletteParameter() {
	QVector2D r = m_image->rangeRatio();
	m_program->setUniformValue("f_rangeMin", r[0]);
	m_program->setUniformValue("f_rangeRatio", r[1]);
	m_program->setUniformValue("f_noHasDataValue", m_image->hasNoDataValue());
	m_program->setUniformValue("f_noDataValue", m_image->noDataValue());
}

void QGLTiledImageItem::setupShader(const QMatrix4x4 &viewProjectionMatrix) {
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

	setPaletteParameter();
	m_program->setUniformValue("color_map", 1);
}

void QGLTiledImageItem::initializeGL() {
	initializeShaders();
	m_initialized = true;

	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	// room for 2 triangles of 3 vertices
	m_vertexBuffer.allocate(2 * 3 * sizeof(QVertex2D));
	m_vertexBuffer.release();

	m_image->bindLUTTexture(COLOR_MAP_TEXTURE_UNIT);
	m_image->releaseLUTTexture(COLOR_MAP_TEXTURE_UNIT);
}

void QGLTiledImageItem::renderTile(const QMatrix4x4 &viewProjectionMatrix,
		QGLTile *tile) {
	m_vertexBuffer.bind();
	const std::vector<QVertex2D> glCoords = tile->coords().glCoords();

	int vCount = 0;
	// first triangle v0, v1, v2
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &glCoords[0],
			sizeof(QVertex2D));
	vCount++;
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &glCoords[1],
			sizeof(QVertex2D));
	vCount++;
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &glCoords[2],
			sizeof(QVertex2D));
	vCount++;

	// second triangle v1, v3, v2
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &glCoords[1],
			sizeof(QVertex2D));
	vCount++;
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &glCoords[3],
			sizeof(QVertex2D));
	vCount++;
	m_vertexBuffer.write(vCount * sizeof(QVertex2D), &glCoords[2],
			sizeof(QVertex2D));
	vCount++;

	m_program->bind();
	setupShader(viewProjectionMatrix);

	m_program->setUniformValue("color_map", COLOR_MAP_TEXTURE_UNIT);
	m_program->setUniformValue("f_tileTexture", IMAGE_TEXTURE_UNIT);
	m_image->bindLUTTexture(COLOR_MAP_TEXTURE_UNIT);
	QOpenGLTexture *texture = tile->getAndBindTexture(IMAGE_TEXTURE_UNIT);


	// draw 2 triangles = 6 vertices starting at offset 0 in the buffer
	glDrawArrays(GL_TRIANGLES, 0, 6);

	// release texture
	texture->release(IMAGE_TEXTURE_UNIT,QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
	m_image->releaseLUTTexture(COLOR_MAP_TEXTURE_UNIT);

	m_vertexBuffer.release();
	m_program->release();
}

void QGLTiledImageItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	if (!m_initialized) {
		initializeGL();
		m_initialized = true;
	}
	glClearStencil(0);
	glClearColor(0, 0, 0, 0);
	glStencilMask(~0);
	glClear(GL_STENCIL_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// program setup

	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}

	//Query to get all the current visible tiles
	std::vector<QGLTileCoord> visibleTiles = m_image->getTilesCoords(exposed);

	//From all the tiles, display the once that are cached
	std::vector<QGLTile*> cachedTiles;
	for (const QGLTileCoord t : visibleTiles) {
		QGLTile *c = m_image->cachedImageTile(t);
		if (c != nullptr)
			renderTile(viewProjectionMatrix, c);
		else
			m_thread->requestTile(t);
	}
}

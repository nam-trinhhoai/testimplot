#include "qglfullcudaimageitem.h"
#include "cudaimagetexturemapper.h"
#include <iostream>
#include <iomanip>
#include <limits>

#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include "iimagepaletteholder.h"
#include "qvertex2D.h"
#include "qglimageitemhelper.h"

#include "cpuimagepaletteholder.h"
#include <QByteArray>


#define COLOR_MAP_TEXTURE_UNIT 1
#define IMAGE_TEXTURE_UNIT 0


QGLFullCUDAImageItem::QGLFullCUDAImageItem(IImagePaletteHolder *image,
		QGraphicsItem *parent,bool applyMask) :
		QAbstractGLGraphicsItem(parent) {
	m_image = image;

	if(m_image == nullptr)
	{
		qDebug()<<" je suis nullll";
	}

	m_worldExtent = image->worldExtent();
	m_initialized = false;

	m_program = new QOpenGLShaderProgram(this);
	m_mapper=new CUDAImageTextureMapper(image,this);




	m_ApplyMask = applyMask;
	setMask();



}

void QGLFullCUDAImageItem::setMask()
{
	if (m_ApplyMask)
	{
		QVector<QPointF> vec = dynamic_cast<GraphEditor_ItemInfo* >(parentItem())->SceneCordinatesPoints();
		QPolygonF item_scenePoints(vec);
		QPolygonF image_scenePoints(m_worldExtent);
		QPolygonF intersectionPoints = item_scenePoints.intersected(image_scenePoints);
		QPolygonF poly;
		CUDAImagePaletteHolder *paletteHolder = dynamic_cast<CUDAImagePaletteHolder *>(m_image);
		foreach(QPointF p, intersectionPoints)
		{
			double i,j;
			paletteHolder->worldToImage(p.x(), p.y(),i,j);
			poly.push_back(QPointF(i,j));
		}
		QByteArray array = applyMaskTexture(parentItem(), m_image->width(), m_image->height(),poly );
		setMaskTexture(m_image->width(), m_image->height(), array,m_image, this);
	}
	else
	{
		QByteArray noMaskArray(1,255);
		setMaskTexture(1, 1, noMaskArray,m_image, this);
	}
}

void QGLFullCUDAImageItem::updateImage(IImagePaletteHolder *image)
{

	m_image = image;
	m_mapper->deleteLater();

	m_mapper=new CUDAImageTextureMapper(image,this);


	setMask();
	m_initializedCorner = false;
}

QGLFullCUDAImageItem::~QGLFullCUDAImageItem() {

}

void QGLFullCUDAImageItem::preInitGL()
{
}
void QGLFullCUDAImageItem::postInitGL()
{

}
void QGLFullCUDAImageItem::initializeShaders() {


	m_program->bind();
	bool done = false;
	do {
		ImageFormats::QColorFormat colorFormat = m_image->colorFormat();
		if (colorFormat == ImageFormats::QColorFormat::RGB_INTERLEAVED|| colorFormat == ImageFormats::QColorFormat::RGBA_INTERLEAVED) {

			if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",":shaders/images/lut_ifrag.glsl"))
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

void QGLFullCUDAImageItem::initializeCornerGL() {


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

	m_initializedCorner = true;
}

void QGLFullCUDAImageItem::initializeGL() {
	texturemapper->bindTexture(MASK_TEXTURE_UNIT);
	texturemapper->releaseTexture(MASK_TEXTURE_UNIT);

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



	m_mapper->bindLUTTexture(COLOR_MAP_TEXTURE_UNIT);
	m_mapper->releaseLUTTexture(COLOR_MAP_TEXTURE_UNIT);

	m_mapper->bindTexture(IMAGE_TEXTURE_UNIT);
	m_mapper->releaseTexture(IMAGE_TEXTURE_UNIT);

	if (texturemapper)
	{
		texturemapper->bindTexture(MASK_TEXTURE_UNIT);
		texturemapper->releaseTexture(MASK_TEXTURE_UNIT);
	}

	postInitGL();
	m_initializedCorner=true;
	m_initialized = true;
}

void QGLFullCUDAImageItem::setPaletteParameter(QOpenGLShaderProgram *program) {
	QVector2D r = m_image->rangeRatio();
	program->setUniformValue("f_rangeMin", r[0]);
	program->setUniformValue("f_rangeRatio", r[1]);
	program->setUniformValue("f_noHasDataValue", m_image->hasNoDataValue());
	program->setUniformValue("f_noDataValue", m_image->noDataValue());
}

void QGLFullCUDAImageItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {

	if(m_image ==nullptr) return;

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);



	if (!m_initialized)
		initializeGL();
	if(!m_initializedCorner)
		initializeCornerGL();

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
	m_program->setUniformValue("mask_Texture", MASK_TEXTURE_UNIT);

	m_mapper->bindLUTTexture(COLOR_MAP_TEXTURE_UNIT);
	m_mapper->bindTexture(IMAGE_TEXTURE_UNIT);


	if (texturemapper)
	{
		texturemapper->bindTexture(MASK_TEXTURE_UNIT);
	}


	// draw 2 triangles = 6 vertices starting at offset 0 in the buffer
	glDrawArrays(GL_TRIANGLES, 0, 6);
	// release texture
	m_mapper->releaseTexture(IMAGE_TEXTURE_UNIT);
	m_mapper->releaseLUTTexture(COLOR_MAP_TEXTURE_UNIT);


	if (texturemapper)
	{
		texturemapper->releaseTexture(MASK_TEXTURE_UNIT);
	}


	m_vertexBuffer.release();
	m_program->release();
}


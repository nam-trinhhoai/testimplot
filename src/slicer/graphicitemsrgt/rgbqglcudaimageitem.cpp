#include "rgbqglcudaimageitem.h"

#include <QPainter>
#include <QOpenGLTexture>
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector2D>
#include <QGraphicsScene>
#include <QGraphicsView>

#include <QOpenGLVertexArrayObject>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObjectFormat>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
//#include <QGLWidget>
#include <QPaintEngine>

#include <qpoint.h>

#include "qvertex2D.h"
#include "cudaimagetexturemapper.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "qglimageitemhelper.h"
#include "imageformats.h"

#define RED_TEXTURE_UNIT 0
#define GREEN_TEXTURE_UNIT 1
#define BLUE_TEXTURE_UNIT 2

#define ISOVALUE_TEXTURE_UNIT 3



RGBQGLCUDAImageItem::RGBQGLCUDAImageItem(
		CUDAImagePaletteHolder *isoSurfaceHolder, CUDARGBImage *holder,
		int defaultExtractionWindow, QGraphicsItem *parent, bool applyMask) :
		QAbstractGLGraphicsItem(parent) {
	m_isoSurfaceImage = isoSurfaceHolder;
	m_image = holder;
	m_initialized = false;
	m_program = new QOpenGLShaderProgram(this);
	m_worldExtent = m_isoSurfaceImage->worldExtent();

	m_isoSurfaceMapper=new CUDAImageTextureMapper(m_isoSurfaceImage,this);

	m_ApplyMask = applyMask;

	if (m_ApplyMask)
	{
		QVector<QPointF> vec = dynamic_cast<GraphEditor_ItemInfo* >(parent)->SceneCordinatesPoints();
		QPolygonF item_scenePoints(vec);
		QPolygonF image_scenePoints(m_worldExtent);
		QPolygonF intersectionPoints = item_scenePoints.intersected(image_scenePoints);
		QPolygonF poly;
		foreach(QPointF p, intersectionPoints)
		{
			double i,j;
			m_isoSurfaceImage->worldToImage(p.x(), p.y(),i,j);
			poly.push_back(QPointF(i,j));
		}
		QByteArray array = applyMaskTexture(parent, m_isoSurfaceImage->width(), m_isoSurfaceImage->height(), poly);
		setMaskTexture(m_isoSurfaceImage->width(), m_isoSurfaceImage->height(), array,m_isoSurfaceImage, this);
		m_array = array;
		for (int j=0; j< m_image->height(); j++)
		{
			for (int i =0; i<m_image->width(); i++)
			{
				if (m_array[j * m_image->width() + i] == 0)
				{
					double value = -9999;
					m_image->get(0)->setValue(i,j, value);
					m_image->get(1)->setValue(i,j, value);
					m_image->get(2)->setValue(i,j, value);
					m_isoSurfaceImage->setValue(i,j, value);
				}
			}
		}
	}
	else
	{
		QByteArray noMaskArray(1,255);
		setMaskTexture(1, 1, noMaskArray,m_isoSurfaceImage, this);
		m_array = noMaskArray;
	}

	m_redMapper=new CUDAImageTextureMapper(m_image->get(0),this);
	m_greenMapper=new CUDAImageTextureMapper(m_image->get(1),this);
	m_blueMapper=new CUDAImageTextureMapper(m_image->get(2),this);

	m_isoSurfaceMapper=new CUDAImageTextureMapper(m_isoSurfaceImage,this);




}


void RGBQGLCUDAImageItem::updateImage(QGraphicsObject* new_img, QGraphicsItem* item)
{
	RGBQGLCUDAImageItem *rgb = dynamic_cast<RGBQGLCUDAImageItem *>(new_img);
	for (int j=0; j< m_image->height(); j++)
	{
		for (int i =0; i<m_image->width(); i++)
		{
			if ((m_array[j * m_image->width() + i] == 0) &&
					(rgb->m_array[j * m_image->width() + i] == 255) )
			{
				double value;
				rgb->m_image->get(0)->valueAt(i,j,value);
				m_image->get(0)->setValue(i,j, value);

				rgb->m_image->get(1)->valueAt(i,j,value);
				m_image->get(1)->setValue(i,j, value);

				rgb->m_image->get(2)->valueAt(i,j,value);
				m_image->get(2)->setValue(i,j, value);

				rgb->m_isoSurfaceImage->valueAt(i,j,value);
				m_isoSurfaceImage->setValue(i,j, value);

				m_array[j * m_image->width() + i] = 255;
			}
			else
			{

			}
		}
	}
	emit m_image->get(0)->dataChanged();
	emit m_image->get(1)->dataChanged();
	emit m_image->get(2)->dataChanged();

	setMaskTexture(m_isoSurfaceImage->width(), m_isoSurfaceImage->height(), m_array,m_isoSurfaceImage, this);

	setParentItem(item);
}




void RGBQGLCUDAImageItem::initShaders() {

	///Test le type
	ImageFormats::QSampleType sampleType = m_image->get(0)->sampleType();
	if(sampleType==ImageFormats::QSampleType::FLOAT32)
	{

		if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",":shaders/images/rgb_elem_frag.glsl"))
			qDebug() << "Failed to initialize shaders";
	}else if (sampleType==ImageFormats::QSampleType::UINT8 &&
			sampleType==ImageFormats::QSampleType::UINT16 &&
			sampleType==ImageFormats::QSampleType::UINT32)
	{

		// UINT64 is ignored because no opengl support for uint64 see https://doc.qt.io/qt-5/qopengltexture.html#PixelType-enum
		if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",
					":shaders/images/rgb_elem_ufrag.glsl"))
				qDebug() << "Failed to initialize shaders";
	} else {

		if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",
					":shaders/images/rgb_elem_ifrag.glsl"))
				qDebug() << "Failed to initialize shaders";
	}
}

RGBQGLCUDAImageItem::~RGBQGLCUDAImageItem() {

	m_vertexBuffer.destroy();
	if (m_program)
		m_program->removeAllShaders();
	m_initialized = false;
}

void RGBQGLCUDAImageItem::initializeGL() {
	m_redMapper->bindTexture(RED_TEXTURE_UNIT);
	m_greenMapper->bindTexture(GREEN_TEXTURE_UNIT);
	m_blueMapper->bindTexture(BLUE_TEXTURE_UNIT);

	m_redMapper->releaseTexture(RED_TEXTURE_UNIT);
	m_greenMapper->releaseTexture(GREEN_TEXTURE_UNIT);
	m_blueMapper->releaseTexture(BLUE_TEXTURE_UNIT);


	m_isoSurfaceMapper->bindTexture(ISOVALUE_TEXTURE_UNIT);
	m_isoSurfaceMapper->releaseTexture(ISOVALUE_TEXTURE_UNIT);


	if (texturemapper)
	{
		texturemapper->bindTexture(MASK_TEXTURE_UNIT);
		texturemapper->releaseTexture(MASK_TEXTURE_UNIT);
	}


	initShaders();

	// vertex buffer initialisation
	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	// room for 2 triangles of 3 vertices
	m_vertexBuffer.allocate(2 * 3 * sizeof(QVertex2D));

	QVertex2D v0, v1, v2, v3;
	QGLImageItemHelper::computeImageCorner(m_isoSurfaceImage, v0, v1, v2, v3);

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

	m_initialized = true;
}

void RGBQGLCUDAImageItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {

	if (!m_initialized)
		initializeGL();

	// enable texturing
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_TEXTURE_1D);
	// enable blending
	glEnable(GL_BLEND);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// a good practice if you have to manage multiple shared context
	// with shared resources: the link is context dependant.
	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}

	//this is not needed here but if we d'ont do this the Iso value texture is not refreshed and the 3D view is not refreshed
	m_isoSurfaceMapper->bindTexture(ISOVALUE_TEXTURE_UNIT);
	m_isoSurfaceMapper->releaseTexture(ISOVALUE_TEXTURE_UNIT);

	// binding the buffer
	m_vertexBuffer.bind();

	// program setup
	m_program->bind();

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

	m_program->setUniformValue("minValueActivated", m_minimumValueActive);
	m_program->setUniformValue("minValue", m_minimumValue);

	m_program->setUniformValue("red_Texture", RED_TEXTURE_UNIT);
	m_program->setUniformValue("green_Texture", GREEN_TEXTURE_UNIT);
	m_program->setUniformValue("blue_Texture", BLUE_TEXTURE_UNIT);
	m_program->setUniformValue("mask_Texture", MASK_TEXTURE_UNIT);


	m_redMapper->bindTexture(RED_TEXTURE_UNIT);
	m_greenMapper->bindTexture(GREEN_TEXTURE_UNIT);
	m_blueMapper->bindTexture(BLUE_TEXTURE_UNIT);

	if (texturemapper)
	{
		texturemapper->bindTexture(MASK_TEXTURE_UNIT);
	}

	glDrawArrays(GL_TRIANGLES, 0, 6);

	m_redMapper->releaseTexture(RED_TEXTURE_UNIT);
	m_greenMapper->releaseTexture(GREEN_TEXTURE_UNIT);
	m_blueMapper->releaseTexture(BLUE_TEXTURE_UNIT);


	if (texturemapper)
	{
		texturemapper->releaseTexture(MASK_TEXTURE_UNIT);
	}


	m_vertexBuffer.release();
	m_program->release();
}

void RGBQGLCUDAImageItem::setPaletteParameter(QOpenGLShaderProgram *program) {

	QVector2D red = m_image->rangeRatio(0);
	program->setUniformValue("red_rangeMin", red[0]);
	program->setUniformValue("red_rangeRatio", red[1]);

	QVector2D green = m_image->rangeRatio(1);
	program->setUniformValue("green_rangeMin", green[0]);
	program->setUniformValue("green_rangeRatio", green[1]);

	QVector2D blue = m_image->rangeRatio(2);
	program->setUniformValue("blue_rangeMin", blue[0]);
	program->setUniformValue("blue_rangeRatio", blue[1]);
}

bool RGBQGLCUDAImageItem::minimumValueActive() const {
	return m_minimumValueActive;
}

void RGBQGLCUDAImageItem::setMinimumValueActive(bool activated) {
	m_minimumValueActive = activated;
}

float RGBQGLCUDAImageItem::minimumValue() const {
	return m_minimumValue;
}

void RGBQGLCUDAImageItem::setMinimumValue(float minValue) {
	m_minimumValue = minValue;
}


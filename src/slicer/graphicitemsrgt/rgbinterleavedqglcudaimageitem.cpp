#include "rgbinterleavedqglcudaimageitem.h"

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
#include "cudargbimagetexturemapper.h"
#include "iimagepaletteholder.h"
#include "cudargbinterleavedimage.h"
#include "qglimageitemhelper.h"
#include "imageformats.h"
#include "cpuimagepaletteholder.h"
#include <QByteArray>

#define RGB_TEXTURE_UNIT 0

#define ISOVALUE_TEXTURE_UNIT 1


RGBInterleavedQGLCUDAImageItem::RGBInterleavedQGLCUDAImageItem(
		IImagePaletteHolder *isoSurfaceHolder, CUDARGBInterleavedImage *holder,
		int defaultExtractionWindow, QGraphicsItem *parent, bool applyMask) :
		QAbstractGLGraphicsItem(parent) {
	m_isoSurfaceImage = isoSurfaceHolder;
	m_image = holder;
	m_initialized = false;

	m_program = new QOpenGLShaderProgram(this);
	m_worldExtent = m_isoSurfaceImage->worldExtent();

	m_rgbMapper=new CUDARGBImageTextureMapper(m_image,this);

	m_isoSurfaceMapper=new CUDAImageTextureMapper(m_isoSurfaceImage,this);


	int image_width = m_isoSurfaceImage->width();
	int image_height = m_isoSurfaceImage->height();


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
		}
		else
		{
			QByteArray noMaskArray(1,255);
			setMaskTexture(1, 1, noMaskArray,m_isoSurfaceImage, this);
		}

}






bool RGBInterleavedQGLCUDAImageItem::InsidePolygon(QVector<QPointF> polygon, QPointF p)
{
	int counter = 0;
	int i;
	double xinters;
	QPointF p1,p2;



	p1 = polygon[0];
	for (i=1;i<=polygon.size();i++) {
		p2 = polygon[i % polygon.size()];
		if (p.y() > qMin(p1.y(),p2.y())) {
			if (p.y() <= qMax(p1.y(),p2.y())) {
				if (p.x() <= qMax(p1.x(),p2.x())) {
					if (p1.y() != p2.y()) {
						xinters = (p.y()-p1.y())*(p2.x()-p1.x())/(p2.y()-p1.y())+p1.x();
						if (p1.x() == p2.x() || p.x() <= xinters)
							counter++;
					}
				}
			}
		}
		p1 = p2;
	}



	if (counter % 2 == 0)
		return false;
	else
		return true;
}


void RGBInterleavedQGLCUDAImageItem::initShaders() {
	///Test le type
	ImageFormats::QSampleType sampleType = m_image->sampleType();
	if(sampleType==ImageFormats::QSampleType::FLOAT32)
	{
		if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",
				":shaders/images/rgbinterleaved_elem_frag.glsl"))
			qDebug() << "Failed to initialize shaders";
	}else if (sampleType==ImageFormats::QSampleType::UINT8 &&
			sampleType==ImageFormats::QSampleType::UINT16 &&
			sampleType==ImageFormats::QSampleType::UINT32)
	{
		// UINT64 is ignored because no opengl support for uint64 see https://doc.qt.io/qt-5/qopengltexture.html#PixelType-enum
		if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",
				":shaders/images/rgbinterleaved_elem_ufrag.glsl"))
			qDebug() << "Failed to initialize shaders";
	} else {
		if (!loadProgram(m_program, ":shaders/images/lut_vert.glsl",
				":shaders/images/rgbinterleaved_elem_ifrag.glsl"))
			qDebug() << "Failed to initialize shaders";
	}
}

RGBInterleavedQGLCUDAImageItem::~RGBInterleavedQGLCUDAImageItem() {

	m_vertexBuffer.destroy();
	if (m_program)
		m_program->removeAllShaders();
	m_initialized = false;
}

void RGBInterleavedQGLCUDAImageItem::initializeGL() {
	m_rgbMapper->bindTexture(RGB_TEXTURE_UNIT);
	m_rgbMapper->releaseTexture(RGB_TEXTURE_UNIT);

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

void RGBInterleavedQGLCUDAImageItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {

	if (!m_initialized)
		initializeGL();

	// enable texturing
	//glEnable(GL_TEXTURE_3D);
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

	m_program->setUniformValue("rgb_Texture", RGB_TEXTURE_UNIT);
	m_program->setUniformValue("mask_Texture", MASK_TEXTURE_UNIT);

	m_rgbMapper->bindTexture(RGB_TEXTURE_UNIT);
	//m_maskTex->bind(MASK_TEXTURE_UNIT);
	texturemapper->bindTexture(MASK_TEXTURE_UNIT);

	if (texturemapper)
	{
		texturemapper->bindTexture(MASK_TEXTURE_UNIT);
	}

	glDrawArrays(GL_TRIANGLES, 0, 6);

	m_rgbMapper->releaseTexture(RGB_TEXTURE_UNIT);
	texturemapper->releaseTexture(MASK_TEXTURE_UNIT);
	//m_maskTex->release(MASK_TEXTURE_UNIT);

	if (texturemapper)
	{
		texturemapper->releaseTexture(MASK_TEXTURE_UNIT);
	}

	m_vertexBuffer.release();
	m_program->release();
}

void RGBInterleavedQGLCUDAImageItem::setPaletteParameter(QOpenGLShaderProgram *program) {

	QVector2D red = m_image->redRangeRatio();
	program->setUniformValue("red_rangeMin", red[0]);
	program->setUniformValue("red_rangeRatio", red[1]);

	QVector2D green = m_image->greenRangeRatio();
	program->setUniformValue("green_rangeMin", green[0]);
	program->setUniformValue("green_rangeRatio", green[1]);

	QVector2D blue = m_image->blueRangeRatio();
	program->setUniformValue("blue_rangeMin", blue[0]);
	program->setUniformValue("blue_rangeRatio", blue[1]);
}

bool RGBInterleavedQGLCUDAImageItem::minimumValueActive() const {
	return m_minimumValueActive;
}

void RGBInterleavedQGLCUDAImageItem::setMinimumValueActive(bool activated) {
	m_minimumValueActive = activated;
}

float RGBInterleavedQGLCUDAImageItem::minimumValue() const {
	return m_minimumValue;
}

void RGBInterleavedQGLCUDAImageItem::setMinimumValue(float minValue) {
	m_minimumValue = minValue;
}


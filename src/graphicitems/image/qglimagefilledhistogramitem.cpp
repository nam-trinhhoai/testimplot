#include "qglimagefilledhistogramitem.h"
#include <QOpenGLShaderProgram>
#include <QGraphicsSceneMouseEvent>
#include <iostream>
#include <QOpenGLTexture>
#include <QGraphicsScene>
#include "igeorefimage.h"
#include "ipaletteholder.h"

QGLImageFilledHistogramItem::QGLImageFilledHistogramItem(IGeorefImage *provider,
		IPaletteHolder *holder,  QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent) {
	m_initialized = false;
	m_worldExtent = IGeorefImage::worldExtent(provider);
	QColor c(Qt::yellow);
	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.5f);

	m_transfoMatrix = provider->imageToWorldTransformation();
	m_image = provider;
	m_holder = holder;

	m_posX = 0;
	m_posY = 0;
	m_visible = false;
	m_horizontalPixelRatio = 0.05f;
	m_verticalPixelRatio = 0.05f;
	m_program = new QOpenGLShaderProgram(this);
}

QGLImageFilledHistogramItem::~QGLImageFilledHistogramItem() {
	m_initialized = false;
}

void QGLImageFilledHistogramItem::initializeGL() {
	initShaders();

	int width = m_image->width();
	int height = m_image->height();

	//Triangle Strip
	m_vertexBufferHorizontal.create();
	m_vertexBufferHorizontal.bind();
	m_vertexBufferHorizontal.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBufferHorizontal.allocate(width * 4 * sizeof(float));

	m_valuesHorizontal.resize(width * 4);
	m_indicesHorizontal.resize(width * 2);
	m_vertexBufferHorizontal.release();

	m_vertexBufferVertical.create();
	m_vertexBufferVertical.bind();
	m_vertexBufferVertical.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBufferVertical.allocate(height * 4 * sizeof(float));

	m_valuesVertical.resize(height * 4);
	m_indicesVertical.resize(height * 2);
	m_vertexBufferVertical.release();

	//Line elements
	m_vertexBufferLineHorizontal.create();
	m_vertexBufferLineHorizontal.bind();
	m_vertexBufferLineHorizontal.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBufferLineHorizontal.allocate(4 * sizeof(float));
	m_vertexBufferLineHorizontal.release();

	m_vertexBufferLineVertical.create();
	m_vertexBufferLineVertical.bind();
	m_vertexBufferLineVertical.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBufferLineVertical.allocate(4 * sizeof(float));
	m_vertexBufferLineVertical.release();

	m_initialized = true;
}
void QGLImageFilledHistogramItem::mouseMove(double worldX, double worldY)
{
	double di, dj;
	m_image->worldToImage(worldX, worldY, di, dj);
	int i = (int) di;
	int j = (int) dj;
	if (i < 0 || j < 0 || i >= m_image->width() || j >= m_image->height()) {
		m_visible = false;
	} else {
		m_posX = i;
		m_posY = j;
		m_visible = true;
		m_needUpdate = true;
	}
	update();
}

void QGLImageFilledHistogramItem::updateHistogramHorizontalGeometry() {
	QVector2D r = m_holder->rangeRatio();
	double scale = m_horizontalPixelRatio * m_image->height();

	bool valid[m_image->width()];
	double vals[m_image->width()];
	m_image->valuesAlongJ(m_posY, valid, vals);
	for (unsigned int i = 0; i < m_image->width(); i++) {
		double val = vals[i];
		if (!valid[i])
			val = r.x();
		val = (val - r.x()) * r.y();
		if (val > 1.0f)
			val = 1.0f;
		if (val < 0)
			val = 0;

		m_valuesHorizontal[4 * i] = i;
		m_valuesHorizontal[4 * i + 1] = m_posY + scale * val;

		m_valuesHorizontal[4 * i + 2] = i;
		m_valuesHorizontal[4 * i + 3] = m_posY;

		m_indicesHorizontal[2 * i] = 2 * i;
		m_indicesHorizontal[2 * i + 1] = 2 * i + 1;
	}
	m_valuesLineHorizontal[0] = 0;
	m_valuesLineHorizontal[1] = m_posY;

	m_valuesLineHorizontal[2] = m_image->width() - 1;
	m_valuesLineHorizontal[3] = m_posY;
}

void QGLImageFilledHistogramItem::updateHistogramVerticalGeometry() {
	QVector2D r = m_holder->rangeRatio();
	double scale = m_verticalPixelRatio * m_image->width();
	bool valid[m_image->height()];
	double vals[m_image->height()];
	m_image->valuesAlongI(m_posX, valid, vals);
	for (unsigned int i = 0; i < m_image->height(); i++) {
		double val = vals[i];
		if (!valid[i])
			val = r.x();

		val = (val - r.x()) * r.y();
		if (val > 1.0f)
			val = 1.0f;
		if (val < 0)
			val = 0;

		m_valuesVertical[4 * i] = m_posX + scale * val;
		m_valuesVertical[4 * i + 1] = i;

		m_valuesVertical[4 * i + 2] = m_posX;
		m_valuesVertical[4 * i + 3] = i;

		m_indicesVertical[2 * i] = 2 * i;
		m_indicesVertical[2 * i + 1] = 2 * i + 1;
	}

	m_valuesLineVertical[0] = m_posX;
	m_valuesLineVertical[1] = 0;

	m_valuesLineVertical[2] = m_posX;
	m_valuesLineVertical[3] = m_image->height() - 1;
}

void QGLImageFilledHistogramItem::initShaders() {
	if(!loadProgram(m_program,":shaders/common/common.vert",":shaders/common/simpleColor.frag"))
			qDebug() << "Failed to initialize shaders";

}

void QGLImageFilledHistogramItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	if (!m_initialized)
		initializeGL();

	if (!m_visible)
		return;

	glClearStencil(0);
	glClearColor(0, 0, 0, 0);
	glStencilMask(~0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}
	if (m_needUpdate) {
		updateHistogramHorizontalGeometry();
		updateHistogramVerticalGeometry();
	}

	m_vertexBufferLineHorizontal.bind();
	if (m_needUpdate)
		m_vertexBufferLineHorizontal.write(0, m_valuesLineHorizontal,
				4 * sizeof(float));
	m_program->bind();
	setupShader(m_program, m_color, viewProjectionMatrix);
	glDrawArrays(GL_LINES, 0, 2);
	m_vertexBufferLineHorizontal.release();
	m_program->release();

	//Vertical straighline pass
	m_vertexBufferLineVertical.bind();
	if (m_needUpdate)
		m_vertexBufferLineVertical.write(0, m_valuesLineVertical,
				4 * sizeof(float));
	m_program->bind();
	setupShader(m_program, m_color, viewProjectionMatrix);
	glDrawArrays(GL_LINES, 0, 2);
	m_vertexBufferLineVertical.release();
	m_program->release();

	m_vertexBufferHorizontal.bind();
	if (m_needUpdate)
		m_vertexBufferHorizontal.write(0, m_valuesHorizontal.data(),
				m_valuesHorizontal.size() * sizeof(float));
	m_program->bind();
	setupShader(m_program, m_color, viewProjectionMatrix);
	glDrawElements( GL_TRIANGLE_STRIP, m_indicesHorizontal.size(),
			GL_UNSIGNED_INT, &m_indicesHorizontal[0]);
	m_vertexBufferHorizontal.release();
	m_program->release();

	m_vertexBufferVertical.bind();
	if (m_needUpdate)
		m_vertexBufferVertical.write(0, m_valuesVertical.data(),
				m_valuesVertical.size() * sizeof(float));
	m_program->bind();
	setupShader(m_program, m_color, viewProjectionMatrix);
	glDrawElements( GL_TRIANGLE_STRIP, m_indicesVertical.size(),
			GL_UNSIGNED_INT, &m_indicesVertical[0]);
	m_vertexBufferVertical.release();
	m_program->release();

	m_needUpdate = false;
}

void QGLImageFilledHistogramItem::setupShader(QOpenGLShaderProgram *program,
		QVector4D color, const QMatrix4x4 &viewProjectionMatrix) {
	program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	program->setUniformValue("color", color);

	program->setUniformValue("transfoMatrix", m_transfoMatrix);
	program->enableAttributeArray("vertexPosition");
	program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2, 0);
}


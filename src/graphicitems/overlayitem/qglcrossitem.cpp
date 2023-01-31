#include "qglcrossitem.h"
#include <QOpenGLShaderProgram>
#include <QGraphicsSceneMouseEvent>
#include <iostream>
#include <QOpenGLTexture>
#include <QGraphicsScene>
#include "igeorefimage.h"
#include "ipaletteholder.h"

QGLCrossItem::QGLCrossItem(const QRectF &worldExtent, QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent) {
	m_initialized = false;
	m_worldExtent = worldExtent;
	QColor c(Qt::red);
	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.5f);
	m_posX = 0;
	m_posY = 0;
	m_needUpdate = false;
	m_transfoMatrix.setToIdentity();
	m_program = new QOpenGLShaderProgram(this);
	memset(m_valuesLine,0,8*sizeof(float));
}

QGLCrossItem::~QGLCrossItem() {

}

void QGLCrossItem::initShaders() {
	if (!loadProgram(m_program, ":shaders/common/common.vert",
			":shaders/common/simpleColor.frag"))
		qDebug() << "Failed to initialize shaders";
}
void QGLCrossItem::setPosition(double worldX, double worldY)
{
	m_posX=worldX;
	m_posY=worldY;
	m_needUpdate = true;
	update();
}
void QGLCrossItem::initializeGL() {
	initShaders();
	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBuffer.allocate(8 * sizeof(float));
	m_vertexBuffer.write(0, m_valuesLine, 8 * sizeof(float));
	m_vertexBuffer.release();
	m_initialized = true;
}

void QGLCrossItem::updateGeometry(const QRectF &exposed) {

	m_valuesLine[0] = exposed.x();
	m_valuesLine[1] = m_posY;

	m_valuesLine[2] = exposed.x() + exposed.width();
	m_valuesLine[3] = m_posY;

	m_valuesLine[4] = m_posX;
	m_valuesLine[5] = exposed.y();

	m_valuesLine[6] = m_posX;
	m_valuesLine[7] = exposed.y() + exposed.height();
}

void QGLCrossItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	if (!m_initialized)
		initializeGL();

	if (m_needUpdate) {
		updateGeometry(exposed);
	}

	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}

	// program setup
	m_program->bind();

	m_vertexBuffer.bind();
	if (m_needUpdate)
		m_vertexBuffer.write(0, m_valuesLine, 8 * sizeof(float));

	setupShader(viewProjectionMatrix, m_transfoMatrix);
	glDrawArrays(GL_LINES, 0, 4);
	m_vertexBuffer.release();
	m_program->release();

	m_needUpdate = false;
}

void QGLCrossItem::setupShader(const QMatrix4x4 &viewProjectionMatrix,
		const QMatrix4x4 &transfo) {
	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", m_color);
	m_program->setUniformValue("transfoMatrix", transfo);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2);
}


#include "qgllineitem.h"
#include <cmath>
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector2D>
#include <QMatrix4x4>
#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include "igeorefimage.h"
#include <iostream>
#include <QGraphicsSceneMouseEvent>
#include "SectionNumText.h"

QGLLineItem::QGLLineItem(const QRectF &imageExtent,
		const IGeorefImage *const imageToWorld, Direction dir,
		QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent), m_extentToWorld(imageToWorld) {
	setAcceptedMouseButtons(Qt::LeftButton);
	m_indexPos = 0.001;
	m_imageExtent = imageExtent;
	m_LineWidth = 1.0;

	m_dir = dir;
	m_initialized = false;
	QColor c;
	if (dir == Direction::HORIZONTAL)
		c = QColor(Qt::cyan);
	else
		c = QColor(155,170,0);

	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.5f);
	m_indexPos = 0.25;
	m_program = new QOpenGLShaderProgram(this);
	m_needUpate = true;
	m_startDrag = false;

	if (m_extentToWorld != nullptr) {
		m_transfo = imageToWorld->imageToWorldTransformation();
		m_worldExtent = IGeorefImage::imageToWorld(imageToWorld,imageExtent);
	} else {
		m_transfo.setToIdentity();
		m_worldExtent = m_imageExtent;
	}
	m_textItem = new SectionNumText(this);
	m_textItem->setVisible(true);

}

void QGLLineItem::setColor(QColor c) {
	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.5f);
	update();
}

QGLLineItem::~QGLLineItem() {
	m_initialized = false;
}

void QGLLineItem::initializeGL() {
	initShaders();

	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBuffer.allocate(4 * sizeof(float));
	m_vertexBuffer.release();
	m_initialized = true;
}

void QGLLineItem::initShaders() {
	if (!loadProgram(m_program, ":shaders/common/common.vert",
			":shaders/common/simpleColor.frag"))
		qDebug() << "Failed to initialize shaders";
}

void QGLLineItem::updatePosition(int position) {
	m_indexPos = position;
	m_needUpate = true;
	update();
}

void QGLLineItem::computeLinePosition() {
	float values[4];
	double x, y;

	if (m_dir == Direction::HORIZONTAL) {
		values[0] = m_imageExtent.x();
		values[1] = m_indexPos;

		values[2] = m_imageExtent.x() + m_imageExtent.width();
		values[3] = m_indexPos;
	} else {
		values[0] = m_indexPos;
		values[1] = m_imageExtent.y();

		values[2] = m_indexPos;
		values[3] = m_imageExtent.y() + m_imageExtent.height();
	}

	if (m_extentToWorld)
	{
		double  wValue0, wValue1, wValue2, wValue3;
		m_extentToWorld->imageToWorld(values[0], values[1], wValue0, wValue1);
		m_extentToWorld->imageToWorld(values[2], values[3], wValue2, wValue3);
		QLineF line(QPointF(wValue0,wValue1), QPointF(wValue2,wValue3));

		m_textItem->setRotation(line.angle());
		m_textItem->setPos(QPointF(wValue0,wValue1));
		m_textItem->setText(QString::number(m_indexPos));
	}

	m_vertexBuffer.write(0, values, 4 * sizeof(float));
}

bool QGLLineItem::computePosition(double worldX, double worldY, int &pos) {
	int i, j;
	if (m_extentToWorld != nullptr) {
		double di, dj;
		m_extentToWorld->worldToImage(worldX, worldY, di, dj);
		i = (int) di;
		j = (int) dj;
		if (!m_imageExtent.contains(i, j))
			return false;
	} else {
		i = (int) worldX;
		j = (int) worldY;
		if (!m_imageExtent.contains(i, j))
			return false;
	}

	if (m_dir == Direction::HORIZONTAL) {
		pos = j;
	} else {
		pos = i;
	}

	return true;
}

void QGLLineItem::mousePressed(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {
	if (keys == Qt::KeyboardModifier::ControlModifier) {
		int pos;
		if (!computePosition(worldX, worldY, pos))
		{
			//return;
		}

		if (m_dir == Direction::HORIZONTAL) {
			if (std::abs(m_indexPos - pos) > 24)
				return;
		} else {
			if (std::abs(m_indexPos - pos) > 56)
				return;
		}
		m_startDrag = true;
	}

}

void QGLLineItem::mouseRelease(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {
	m_startDrag = false;
	emit positionChanged((int) m_indexPos);
}

void QGLLineItem::mouseMoved(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {

	int pos;
	bool positionInImage = false;
	bool CursorNearLine = false;

	if (computePosition(worldX, worldY, pos))
	{
		int diff = std::abs(m_indexPos - pos);
		if ( diff < 36)
		{
			CursorNearLine = true;
		}
		positionInImage = true;
	}

	if (!m_startDrag)
	{
		if (!CursorNearLine)
		{
			m_LineWidth = 1.0;
		}
		else
		{
			m_LineWidth = 4.0;
		}
		return;
	}
	else
	{
		m_LineWidth = 4.0;
	}

	if (!positionInImage)
	{
		positionChanged((int) m_indexPos);
		return;
	}

	m_indexPos = pos;
	m_textItem->setText(QString::number(m_indexPos));
	m_needUpate = true;
	update();
}

void QGLLineItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {

	if (!m_initialized)
		initializeGL();

	if (!m_program->isLinked()) {
		if (!m_program->link()) {
			qDebug() << "Shader not linked";
			return;
		}
	}

	// program setup
	m_program->bind();
	m_vertexBuffer.bind();
	if (m_needUpate) {
		computeLinePosition();
		m_needUpate = false;
	}

	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", m_color);

	m_program->setUniformValue("transfoMatrix", m_transfo);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2);

	glLineWidth(m_LineWidth);
	glDrawArrays(GL_LINES, 0, 2);

	m_vertexBuffer.release();
	m_program->release();
}

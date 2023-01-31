#include "qglpointclouditem.h"

#include <QPainter>
#include <QOpenGLTexture>
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector2D>
#include <QPaintEngine>
#include <QStyleOptionGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>
//#include <QGLContext>
#include <QOpenGLWidget>
#include <QMatrix4x4>

#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>

#include <iostream>

QGLPointCloudItem::QGLPointCloudItem(const QVector<QVector2D> & vertices,const QRectF &worldExtent,QGraphicsItem *parent) : QAbstractGLGraphicsItem(parent) {
	m_color = Qt::magenta;
	m_worldExtent = worldExtent;

	m_vertices.reserve(vertices.size()*2);
	for(unsigned int i=0;i<vertices.size();i++)
	{
		m_vertices.push_back((float)vertices[i].x());
		m_vertices.push_back((float)vertices[i].y());
	}

	m_transfoMatrix.setToIdentity();
	m_initialized=false;

	m_pointSize=5.0f;
	m_program = new QOpenGLShaderProgram(this);
}
void QGLPointCloudItem::setPointSize(float size)
{
	m_pointSize=size;
	update();
}

float QGLPointCloudItem::pointSize() const
{
	return m_pointSize;
}


QColor QGLPointCloudItem::pointColor()const
{
	return m_color;
}

void QGLPointCloudItem::setPointColor(QColor c)
{
	m_color=c;
	update();
}

float QGLPointCloudItem::opacity()const
{
	return m_opacity;
}
void QGLPointCloudItem::setOpacity(float value)
{
	m_opacity=value;
	update();
}

QGLPointCloudItem::~QGLPointCloudItem() {
}

void QGLPointCloudItem::initShaders() {
	if(!loadProgram(m_program,":shaders/common/point.vert",":shaders/common/simpleColor.frag"))
			qDebug() << "Failed to initialize shaders";
}

void QGLPointCloudItem::initBuffers()
{
	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_vertexBuffer.allocate(m_vertices.size() * sizeof(float));
	m_vertexBuffer.write(0, m_vertices.data(), m_vertices.size() * sizeof(float));
	m_vertexBuffer.release();
}

void QGLPointCloudItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,const QRectF &exposed,int width, int height, int dpiX,int dpiY) {
	if(!m_initialized)
	{
		initShaders();
		initBuffers();
		m_initialized=true;
	}
	glClearStencil(0);
	glClearColor(0, 0, 0, 0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	if (!m_program->isLinked()) {
		if(!m_program->link())
			return;
	}

	// program setup
	m_program->bind();
	m_vertexBuffer.bind();

	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", QVector4D(m_color.redF(), m_color.greenF(), m_color.blueF(), m_opacity));

	m_program->setUniformValue("transfoMatrix", m_transfoMatrix);

	m_program->setUniformValue("pointsize", m_pointSize);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2,0);

	glDrawArrays(GL_POINTS,0, m_vertices.size() );

	m_vertexBuffer.release();
	m_program->release();
}


#include "qglpolylineitem.h"

#include <QPainter>
#include <QOpenGLTexture>
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector2D>
#include <QPaintEngine>
#include <QStyleOptionGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>
//#include <QOpenGLContext>
#include <QOpenGLWidget>
#include <QMatrix4x4>

#include <QOpenGLContext>
#include <QOpenGLShaderProgram>

#include <iostream>

QGLPolylineItem::QGLPolylineItem(const QVector<QVector2D> & vertices,const QRectF &worldExtent,QGraphicsItem *parent) : QAbstractGLGraphicsItem(parent) {
	m_lineColor= Qt::red;
	m_lineWidth=1.0f;
	m_opacity=1.0f;

	m_worldExtent = worldExtent;

	m_vertices.reserve(vertices.size()*2);
	for(unsigned int i=0;i<vertices.size();i++)
	{
		m_vertices.push_back((float)vertices[i].x());
		m_vertices.push_back((float)vertices[i].y());
	}

	m_program = new QOpenGLShaderProgram(this);
	m_transfoMatrix.setToIdentity();
	m_initialized=false;
}

float QGLPolylineItem::lineWidth()const
{
	return m_lineWidth;
}

void QGLPolylineItem::setLineWidth(float value)
{
	m_lineWidth=value;
	update();
}

float QGLPolylineItem::opacity()const
{
	return m_opacity;
}
void QGLPolylineItem::setOpacity(float value)
{
	m_opacity=value;
	update();
}

QColor QGLPolylineItem::lineColor()const
{
	return m_lineColor;
}

void QGLPolylineItem::setLineColor(QColor c)
{
	m_lineColor=c;
	update();
}

QGLPolylineItem::~QGLPolylineItem() {

}

void QGLPolylineItem::initShaders() {
	if(!loadProgram(m_program,":shaders/common/common.vert",":shaders/common/simpleColor.frag"))
			qDebug() << "Failed to initialize shaders";
}

void QGLPolylineItem::initBuffers()
{
	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_vertexBuffer.allocate(m_vertices.size() * sizeof(float));
	m_vertexBuffer.write(0, m_vertices.data(), m_vertices.size() * sizeof(float));
	m_vertexBuffer.release();
}

void QGLPolylineItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,const QRectF &exposed,int width, int height, int dpiX,int dpiY) {
	if(!m_initialized)
	{
		initShaders();
		initBuffers();
		m_initialized=true;
	}
	glClearStencil(0);
	glClearColor(0, 0, 0, 0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (!m_program->isLinked()) {
		if(!m_program->link())
			return;
	}

	// program setup
	m_program->bind();
	m_vertexBuffer.bind();

	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", QVector4D(m_lineColor.redF(),m_lineColor.greenF(),m_lineColor.blueF(),m_opacity));

	m_program->setUniformValue("transfoMatrix", m_transfoMatrix);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2,0);

	glLineWidth(m_lineWidth);
	glDrawArrays( GL_LINE_STRIP, 0, m_vertices.size()/2);

	m_vertexBuffer.release();
	m_program->release();

	glLineWidth(1.0f);
}


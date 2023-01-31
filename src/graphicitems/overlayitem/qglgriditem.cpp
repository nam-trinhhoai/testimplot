#include "qglgriditem.h"

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
#include <QMatrix2x2>

#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <cmath>
using namespace std;

QGLGridItem::QGLGridItem(const QRectF &worldExtent, QGraphicsItem *parent) :
																		QAbstractGLGraphicsItem(parent) {
	m_initialized = false;

	QColor c(Qt::gray);
	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.25f);

	m_worldExtent = worldExtent;
	m_tickRatio = 0.1;
	m_transfoMatrix.setToIdentity();
	m_program = new QOpenGLShaderProgram(this);
}

QGLGridItem::~QGLGridItem() {
	m_initialized = false;
}

void QGLGridItem::UpdateRatio(float f){
	m_tickRatio = f;
}

void QGLGridItem::initializeGL() {
	initShaders();

	float intervalH = m_tickRatio * m_worldExtent.height();
	float intervalW = m_tickRatio * m_worldExtent.width();

	m_numMaxPoints = 2*(m_worldExtent.height() / intervalH + m_worldExtent.width() / intervalW + 2);
	m_numMaxPoints*= 4;
	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBuffer.allocate(2 * m_numMaxPoints * sizeof(float));
	m_vertexBuffer.release();
	m_initialized = true;
}


int QGLGridItem::updateInternalBuffer(const QRectF &exposed,int iWidth,int iHeight) {

	int lineCount;
	float height = ceil(exposed.height());
	float width = ceil(exposed.width());
	std::vector<float> vertices;
	m_vertexBuffer.bind();

	vertices.reserve(2*m_numMaxPoints);

	float pix,rangeMin,rangeMax,rangeSize;
	int w = ceil(m_worldExtent.width()/10)*10 - 1;
	float y = ceil(exposed.y());
	float h = ceil(exposed.height());
	float x = ceil(exposed.x());

	double xmax = x + width;
	double ymax = y + height;

	for(int vertical = 0;vertical<2;vertical++){

		if (vertical == 0) {
			pix  = iWidth;
			rangeMin = exposed.x();
			rangeMax = exposed.width() + rangeMin;
			rangeSize = rangeMax - rangeMin;
		}else{
			pix  = iHeight;
			rangeMin = exposed.y();
			rangeMax = exposed.height() + rangeMin;
			rangeSize = rangeMax - rangeMin;
		}
		const int idx0          = 0;
		const int nMinor        = 10;
		const int nMajor        = nvMax(2, (int)NV_ROUND(pix / (vertical ? 300.0f : 400.0f)));
		const double nice_range = NiceNum(rangeSize* 0.99, false);
		const double interval   = NiceNum(nice_range / (nMajor - 1), true);
		const double graphmin   = floor(rangeMin / interval) * interval;
		const double graphmax   = ceil(rangeMax/ interval) * interval;
		bool first_major_set    = false;
		int  first_major_idx    = 0;

		for (double major = graphmin; major < graphmax + 0.5 * interval; major += interval){

			if (major - interval < 0 && major + interval > 0){
				major = 0;
			}

			if (major >= rangeMin && major <= rangeMax){
				if (!first_major_set){
					first_major_set = true;
				}
				if (vertical == 0){
					vertices.push_back(major);
					vertices.push_back(y);
					vertices.push_back(major);
					vertices.push_back(ymax);
				}else{
					vertices.push_back(x);
					vertices.push_back(major);
					vertices.push_back(xmax);
					vertices.push_back(major);
				}
			}

			for (int i = 1; i < nMinor; ++i){
				double minor = major + i * interval / nMinor;

				if (minor >= rangeMin && minor <= rangeMax){
					if (vertical == 0){
						vertices.push_back(minor);
						vertices.push_back(y);
						vertices.push_back(minor);
						vertices.push_back(ymax);
					}else{
						vertices.push_back(x);
						vertices.push_back(minor);
						vertices.push_back(xmax);
						vertices.push_back(minor);
					}
				}
			}
		}

		if (vertical == 0){
			vertices.push_back(xmax);
			vertices.push_back(y);
			vertices.push_back(xmax);
			vertices.push_back(ymax);
		}
	}

	vertices.push_back(x);
	vertices.push_back(ymax);
	vertices.push_back(xmax);
	vertices.push_back(ymax);

	lineCount = std::min(m_numMaxPoints, (int) (vertices.size() / 2));
	m_vertexBuffer.write(0, vertices.data(), 2 * lineCount * sizeof(float));
	return lineCount;
}

void QGLGridItem::initShaders() {
	if (!loadProgram(m_program, ":shaders/common/common.vert",
			":shaders/common/simpleColor.frag"))
		qDebug() << "Failed to initialize shaders";
}

void QGLGridItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,const QRectF &exposed, int width, int height, int dpiX, int dpiY) {

	glClearStencil(0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (!m_initialized)
		initializeGL();

	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}

	// program setup
	m_program->bind();

	int lineCount = updateInternalBuffer(exposed,width, height);
	setupShader(viewProjectionMatrix, m_transfoMatrix);
	glDrawArrays(  GL_LINES, 0, lineCount);

	m_vertexBuffer.release();
	m_program->release();
}

void QGLGridItem::setupShader(const QMatrix4x4 &viewProjectionMatrix,
		const QMatrix4x4 &transfo) {
	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", m_color);
	m_program->setUniformValue("transfoMatrix", transfo);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2);
}

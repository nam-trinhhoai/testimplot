#include "qglpolygonitem.h"

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
#include <QOpenGLExtraFunctions>
#include <cmath>
#include <iostream>
//#define GL_GLEXT_PROTOTYPES
#include <qopenglext.h>

QGLPolygonItem::QGLPolygonItem(const QVector<QVector2D> &vertices,
		const QRectF &worldExtent, QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent) {
	m_worldExtent = worldExtent;

	int size = vertices.size();
	m_numVertices = size;
	m_pathCommands = new GLubyte[size + 2];
	m_pathCoords = new GLfloat[2 * (size + 1)];
	for (int i = 0; i < size; i++) {
		m_pathCoords[2 * i] = vertices[i].x();
		m_pathCoords[2 * i + 1] = vertices[i].y();

//		std::cout<<m_pathCoords[2*i]<<"\t"<<m_pathCoords[2*i+1]<<std::endl;
		if (i == 0)
			m_pathCommands[i] = GL_MOVE_TO_NV;
		else
			m_pathCommands[i] = GL_LINE_TO_NV;
	}

	m_pathCommands[size] = GL_LINE_TO_NV;

	//Close the path (we force it!)
	m_pathCoords[2 * size] = m_pathCoords[0];
	m_pathCoords[2 * size + 1] = m_pathCoords[1];

	m_pathCommands[size + 1] = GL_CLOSE_PATH_NV;

	m_initialized = false;

	m_lineWidth = 1.0f;

	m_outlineColor = Qt::cyan;
	m_interiorColor = Qt::blue;
	m_opacity = 1.0f;
}
float QGLPolygonItem::lineWidth() const {
	return m_lineWidth;
}

void QGLPolygonItem::setLineWidth(float value) {
	m_lineWidth = value;
	update();
}

float QGLPolygonItem::opacity() const {
	return m_opacity;
}
void QGLPolygonItem::setOpacity(float value) {
	m_opacity = value;
	update();
}

QColor QGLPolygonItem::outlineColor() const {
	return m_outlineColor;
}
void QGLPolygonItem::setOutlineColor(QColor c) {
	m_outlineColor = c;
	update();
}

QColor QGLPolygonItem::interiorColor() const {
	return m_interiorColor;
}
void QGLPolygonItem::setInteriorColor(QColor c) {
	m_interiorColor = c;
	update();
}

QGLPolygonItem::~QGLPolygonItem() {
	delete[] m_pathCommands;
	delete[] m_pathCoords;
}
//void makeGradient3x3Texture(GLuint texobj)
//{
//  GLfloat pixels[3][3][3];
//  int i, j;
//
//  for (i=0; i<3; i++) {
//    for (j=0; j<3; j++) {
//      pixels[i][j][0] = i/3.0;
//      pixels[i][j][1] = 1-j/3.0;
//      pixels[i][j][2] = 1;
//    }
//  }
//
//  pixels[1][1][2] = 0;  /* Force blue to zero for the middle texel. */
//
//  glBindTexture(GL_TEXTURE_2D, texobj);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 3,3,0, GL_RGB, GL_FLOAT, pixels);
//}

void QGLPolygonItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	if (!m_initialized) {
		m_nvPathFuncs.reset(new QOpenGLExtension_NV_path_rendering);
		m_nvPathFuncs->initializeOpenGLFunctions();
		m_initialized = true;
	}
	glClearStencil(0);
	glClearColor(0, 0, 0, 0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

//	if (m_flippedVertically) {
//		glTranslatef(0, 2 * exposed.y() + exposed.height(), 0);
//		glScalef(1, -1, 1);
//	}

	float lineWidth = m_lineWidth * (float) exposed.width() / width;

	GLuint pathObj = 42;
	m_nvPathFuncs->glPathCommandsNV(pathObj, m_numVertices + 2, m_pathCommands,
			2 * (m_numVertices + 1), GL_FLOAT, m_pathCoords);
	m_nvPathFuncs->glStencilFillPathNV(pathObj, GL_COUNT_UP_NV, 0x1F);

	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
	glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
	GLfloat data[2][3] = { { 1,0,0 },    // s = 1*x + 0*y + 0
	                         { 0,1,0 } };  // t = 0*x + 1*y + 0
	m_nvPathFuncs->glPathTexGenNV(GL_TEXTURE0, GL_COUNT_UP_NV, 2, &data[0][0]);
	glColor4f(m_interiorColor.redF(), m_interiorColor.greenF(),
			m_interiorColor.blueF(), m_opacity);
	m_nvPathFuncs->glCoverFillPathNV(pathObj, GL_BOUNDING_BOX_NV);

	m_nvPathFuncs->glPathParameteriNV(pathObj, GL_PATH_JOIN_STYLE_NV,
	GL_ROUND_NV);
	m_nvPathFuncs->glPathParameterfNV(pathObj, GL_PATH_STROKE_WIDTH_NV,
			lineWidth);	//Expressed in ground coordinate!

	m_nvPathFuncs->glStencilStrokePathNV(pathObj, 0x1, ~0);
	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
	glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

	glColor4f(m_outlineColor.redF(), m_outlineColor.greenF(),
			m_outlineColor.blueF(), m_opacity); // yellow
	m_nvPathFuncs->glCoverStrokePathNV(pathObj, GL_CONVEX_HULL_NV);

//	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
//    glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
//    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
//    GLuint TEXTURE_GRADIENT_3X3=1;
//    makeGradient3x3Texture(TEXTURE_GRADIENT_3X3);

	glDisable(GL_STENCIL_TEST);

	m_nvPathFuncs->glDeletePathsNV(pathObj, m_numVertices + 2);
}


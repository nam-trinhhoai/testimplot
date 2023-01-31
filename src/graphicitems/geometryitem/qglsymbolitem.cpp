#include "qglsymbolitem.h"

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

QGLSymbolItem::QGLSymbolItem(const QPointF & vertice, const char symbol,const QRectF &worldExtent,QGraphicsItem *parent) : QAbstractGLGraphicsItem(parent) {
	m_worldExtent = worldExtent;
	m_point=vertice;
	m_initialized=false;
	m_size=100.0f;
	m_color=Qt::red;
	m_symbol=symbol;

}
float QGLSymbolItem::size()const
{
	return m_size;
}

void QGLSymbolItem::setSize(float value)
{
	m_size=value;
	update();
}


QColor QGLSymbolItem::color()const
{
	return m_color;
}
void QGLSymbolItem::setColor(QColor c)
{
	m_color=c;
	update();
}


QGLSymbolItem::~QGLSymbolItem() {
}


void QGLSymbolItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,const QRectF &exposed,int width, int height, int dpiX,int dpiY) {
	if(!m_initialized)
	{
		m_nvPathFuncs.reset(new QOpenGLExtension_NV_path_rendering);
		m_nvPathFuncs->initializeOpenGLFunctions();
		m_initialized=true;
	}
	glClearStencil(0);
	glClearColor(0, 0, 0, 0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	const char symbol=m_symbol;

	const GLfloat emScale = 2048;  // match TrueType convention
	const int numChars = 256;  // ISO/IEC 8859-1 8-bit character range
	GLuint glyphBase = m_nvPathFuncs->glGenPathsNV(numChars+1);
	m_nvPathFuncs->glPathGlyphRangeNV(glyphBase,
					   GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV,
					   0, numChars,
					   GL_USE_MISSING_GLYPH_NV, ~0, emScale);

	GLfloat xyMinMax[4];
	m_nvPathFuncs->glGetPathMetricRangeNV(
			GL_FONT_X_MIN_BOUNDS_BIT_NV | GL_FONT_X_MAX_BOUNDS_BIT_NV
					| GL_FONT_Y_MIN_BOUNDS_BIT_NV | GL_FONT_Y_MAX_BOUNDS_BIT_NV,
			glyphBase, 1, 4 * sizeof(GLfloat), xyMinMax);


	double fontW=xyMinMax[1]-xyMinMax[0];
	double fontH=xyMinMax[3]-xyMinMax[2];

	double fScale=m_size*exposed.width()/width*std::max(1/fontW,1/fontH);

	glTranslatef( m_point.x(),m_point.y(), 0.0f );

	glScalef( fScale, fScale, fScale );
	glTranslatef( -fontW/2,fontH/2, 0.0f );

	GLfloat xtranslate[2];
	xtranslate[0]=0;
	m_nvPathFuncs->glGetPathSpacingNV(GL_ACCUM_ADJACENT_PAIRS_NV,
			1, GL_UNSIGNED_BYTE, &symbol,
							   glyphBase,
							   1.0f, 1.0f,
							   GL_TRANSLATE_X_NV,
							   &xtranslate[1]);

	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
	glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

	glColor3f(m_color.redF(),m_color.greenF(),m_color.blueF());

	m_nvPathFuncs->glStencilFillPathInstancedNV(1, GL_UNSIGNED_BYTE,  &symbol,
								 glyphBase,
								 GL_PATH_FILL_MODE_NV, 0xFF,
								 GL_TRANSLATE_X_NV, xtranslate);

	m_nvPathFuncs->glCoverFillPathInstancedNV(1, GL_UNSIGNED_BYTE,  &symbol,
									   glyphBase,
									   GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
									   GL_TRANSLATE_X_NV, xtranslate);

	glDisable(GL_STENCIL_TEST);
	m_nvPathFuncs->glDeletePathsNV(glyphBase, numChars+1);

}


//#define GL_GLEXT_PROTOTYPES  //Warning ici (en haut ) car qopenglext est deja inclus



#include "qglcolorbar.h"

#include <QPainter>
#include <QOpenGLTexture>
#include <QTransform>
#include <QOpenGLBuffer>
//#include <QGLContext>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObjectFormat>
#include <QOpenGLShaderProgram>
#include <QGraphicsScene>
#include <QOpenGLPixelTransferOptions>



//#include "qopenglext.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include "qvertex2D.h"
#include "colortableregistry.h"
#include "texturehelper.h"

//#include <GL/freeglut.h>


#define COLOR_MAP_TEXTURE_UNIT 5


QGLColorBar::QGLColorBar(const QRectF &worldExtent, QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent), m_vertexColorTableOverlay(
				new QOpenGLBuffer), m_patchColorMapProgram(
				new QOpenGLShaderProgram(this)) {
	m_initialized = false;

	m_lookupTable = ColorTableRegistry::DEFAULT();
	m_colorMapTexture = nullptr;

	m_range.setX(0.0);
	m_range.setY(1.0);

	m_opacity = 1.0;

	m_worldExtent = worldExtent;
}

void QGLColorBar::setRange(const QVector2D &range) {
	m_range = range;
	update();
}

void QGLColorBar::setOpacity(float value) {
	m_opacity = value;
	update();
}

void QGLColorBar::setLookupTable(const LookupTable &table) {
	m_lookupTable = table;
	generateLUTTexture(true);
	update();
}

QGLColorBar::~QGLColorBar() {
	m_initialized = false;
}

QOpenGLTexture* QGLColorBar::generateLUTTexture(bool force) {
	if (m_colorMapTexture != nullptr && !force) {
		return m_colorMapTexture;
	}

	if (m_colorMapTexture != nullptr && force) {
		m_colorMapTexture->destroy();
	}
	m_colorMapTexture=TextureHelper::generateLUTTexture(m_lookupTable);
	return m_colorMapTexture;
}

void QGLColorBar::initializeGL() {
	if (!m_vertexColorTableOverlay->create()) {
		qDebug() << "Ooops!";
		return;
	}
	m_vertexColorTableOverlay->setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexColorTableOverlay->bind();
	// room for 2 triangles of 3 vertices
	m_vertexColorTableOverlay->allocate(4 * sizeof(QVertex2D));
	m_vertexColorTableOverlay->release();

	if(!loadProgram(m_patchColorMapProgram,":/shaders/colorscale/displayLUT.vert",":/shaders/colorscale/displayLUT.frag"))
			qDebug() << "Failed to initialize shaders";

	m_nvPathFuncs.reset(new QOpenGLExtension_NV_path_rendering);
	m_nvPathFuncs->initializeOpenGLFunctions();
	m_initialized = true;
}

void QGLColorBar::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	if (!m_initialized) {
		initializeGL();
	}

	glClearStencil(0);
	glClearColor(0, 0, 0, 0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// draw 2D here
	//Draw in ortho view. As we don't set the projection matrix, coordinate are expressed in wiew screen
	if (!m_patchColorMapProgram->isLinked())
		m_patchColorMapProgram->link();

	m_patchColorMapProgram->bind();
	m_vertexColorTableOverlay->bind();

	// setup of the program attributes
	int pos = 0, count;
	// positions : 2 floats
	count = 2;
	m_patchColorMapProgram->enableAttributeArray("vertexPosition");
	m_patchColorMapProgram->setAttributeBuffer("vertexPosition", GL_FLOAT, pos,
			count, sizeof(QVertex2D));
	pos += count * sizeof(float);

	// texture coordinates : 2 floats
	count = 2;
	m_patchColorMapProgram->enableAttributeArray("textureCoordinates");
	m_patchColorMapProgram->setAttributeBuffer("textureCoordinates", GL_FLOAT,
			pos, count, sizeof(QVertex2D));
	pos += count * sizeof(float);


	m_patchColorMapProgram->setUniformValue("f_colorMap", COLOR_MAP_TEXTURE_UNIT);
	QOpenGLTexture *colorMapTexture = generateLUTTexture(false);
	colorMapTexture->bind(COLOR_MAP_TEXTURE_UNIT,QOpenGLTexture::TextureUnitReset::ResetTextureUnit);

	float sizeX = 40;

	float xOffset = 20.0f;
	float yOffset = 50.0f;

	float w = width;
	float h = height;

	float xorig = 2 * (width - sizeX - xOffset) / w - 1;
	float yorig = 2 * yOffset / h - 1;

	float xmax = 2 * (width - xOffset) / w - 1;
	float ymax = 2 * (height - yOffset) / h - 1;

	QRectF rect(xorig, yorig, xmax - xorig, ymax - yorig);

	QVertex2D v0;
	v0.position = QVector2D(rect.bottomLeft());
	v0.coords = QVector2D(0, 1.0f);

	QVertex2D v1;
	v1.position = QVector2D(rect.topLeft());
	v1.coords = QVector2D(0, 0);

	QVertex2D v2;
	v2.position = QVector2D(rect.bottomRight());
	v2.coords = QVector2D(1.0f, 1.0f);

	QVertex2D v3;
	v3.position = QVector2D(rect.topRight());
	v3.coords = QVector2D(1.0f, 0);

	int vCount = 0;
	m_vertexColorTableOverlay->write(vCount * sizeof(QVertex2D), &v0,
			sizeof(QVertex2D));
	vCount++;
	m_vertexColorTableOverlay->write(vCount * sizeof(QVertex2D), &v1,
			sizeof(QVertex2D));
	vCount++;
	m_vertexColorTableOverlay->write(vCount * sizeof(QVertex2D), &v3,
			sizeof(QVertex2D));
	vCount++;
	m_vertexColorTableOverlay->write(vCount * sizeof(QVertex2D), &v2,
			sizeof(QVertex2D));
	vCount++;

	//glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	glDrawArrays(GL_QUADS, 0, 4);
	//glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);
	colorMapTexture->release(COLOR_MAP_TEXTURE_UNIT,QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
	m_vertexColorTableOverlay->release();
	m_patchColorMapProgram->release();



	const GLfloat emScale = 2048;  // match TrueType convention
	const int numChars = 256;  // ISO/IEC 8859-1 8-bit character range

	GLuint glyphBase = m_nvPathFuncs->glGenPathsNV(numChars + 1);
	m_nvPathFuncs->glPathGlyphRangeNV(glyphBase,
	GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV, 0, numChars,
	GL_USE_MISSING_GLYPH_NV, ~0, emScale);

	GLfloat xyMinMax[4];
	m_nvPathFuncs->glGetPathMetricRangeNV(
			GL_FONT_X_MIN_BOUNDS_BIT_NV | GL_FONT_X_MAX_BOUNDS_BIT_NV
					| GL_FONT_Y_MIN_BOUNDS_BIT_NV | GL_FONT_Y_MAX_BOUNDS_BIT_NV,
			glyphBase, 1, 4 * sizeof(GLfloat), xyMinMax);

	double fontW = xyMinMax[1] - xyMinMax[0];
	double fontH = xyMinMax[3] - xyMinMax[2];

	float textPixelSize = 7.0f;
	QColor textColor = Qt::white;

	double fScale = textPixelSize * std::max(1 / fontW, 1 / fontH);

	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) << m_range.y();
	std::string max = ss.str();
	ss.str("");
	ss << std::fixed << std::setprecision(2) << m_range.x();
	std::string min = ss.str();

	glColor3f(textColor.redF(), textColor.greenF(), textColor.blueF());
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
	glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

	QMatrix4x4 saved;
	glGetFloatv( GL_MODELVIEW_MATRIX, saved.data());

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(width - sizeX - xOffset, yOffset / 2, 0.0f);
	glScalef(fScale, -fScale, fScale);
	drawText(max, glyphBase);
	glLoadIdentity();
	glTranslatef(width - sizeX - xOffset, height - yOffset / 2, 0.0f);
	glScalef(fScale, -fScale, fScale);
	drawText(min, glyphBase);

	glDisable(GL_STENCIL_TEST);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(saved.data());
	m_nvPathFuncs->glDeletePathsNV(glyphBase, numChars + 1);
}


void QGLColorBar::drawText(const std::string &text, GLuint glyphBase) {
	GLfloat xtranslate[text.length() + 1];
	xtranslate[0] = 0;
	m_nvPathFuncs->glGetPathSpacingNV(GL_ACCUM_ADJACENT_PAIRS_NV, text.length(),
	GL_UNSIGNED_BYTE, text.c_str(), glyphBase, 1.0f, 1.0f,
	GL_TRANSLATE_X_NV, &xtranslate[1]);

	m_nvPathFuncs->glStencilFillPathInstancedNV(text.length(), GL_UNSIGNED_BYTE,
			text.c_str(), glyphBase,
			GL_PATH_FILL_MODE_NV, 0xFF,
			GL_TRANSLATE_X_NV, xtranslate);

	m_nvPathFuncs->glCoverFillPathInstancedNV(text.length(), GL_UNSIGNED_BYTE,
			text.c_str(), glyphBase,
			GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
			GL_TRANSLATE_X_NV, xtranslate);
}


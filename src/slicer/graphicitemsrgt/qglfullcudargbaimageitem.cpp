#include "qglfullcudargbaimageitem.h"
#include "cudaimagetexturemapper.h"
#include <iostream>
#include <iomanip>
#include <limits>

#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include "cudaimagepaletteholder.h"
#include "qvertex2D.h"
#include "qglimageitemhelper.h"

#define RED_IMAGE_TEXTURE_UNIT 0
#define GREEN_IMAGE_TEXTURE_UNIT 1
#define BLUE_IMAGE_TEXTURE_UNIT 2
#define ALPHA_IMAGE_TEXTURE_UNIT 3

QGLFullCUDARgbaImageItem::QGLFullCUDARgbaImageItem(CUDAImagePaletteHolder *red,
		CUDAImagePaletteHolder *green, CUDAImagePaletteHolder *blue,
		CUDAImagePaletteHolder *alpha, RgbDataset::Mode mode, QGraphicsItem *parent) :
		QAbstractGLGraphicsItem(parent) {
	m_red = red;
	m_green = green;
	m_blue = blue;
	m_alpha = alpha;
	m_opacity = 1.0; // m_opacity is used if m_alpha == nullptr
	m_worldExtent = red->worldExtent();
	m_initialized = false;
	m_alphaMode = mode;
	m_radiusAlpha = 0;

	m_program = new QOpenGLShaderProgram(this);
	m_mapperRed = new CUDAImageTextureMapper(m_red,this);
	m_mapperGreen = new CUDAImageTextureMapper(m_green,this);
	m_mapperBlue = new CUDAImageTextureMapper(m_blue,this);
	if (m_alpha!=nullptr) {
		m_mapperAlpha = new CUDAImageTextureMapper(m_alpha,this);
	} else {
		m_mapperAlpha = nullptr;
	}
}

QGLFullCUDARgbaImageItem::~QGLFullCUDARgbaImageItem() {

}

void QGLFullCUDARgbaImageItem::preInitGL()
{
}
void QGLFullCUDARgbaImageItem::postInitGL()
{

}

QString QGLFullCUDARgbaImageItem::samplerType(ImageFormats::QSampleType type) {
	QString sampler;
	if (type == ImageFormats::QSampleType::UINT8
			|| type == ImageFormats::QSampleType::UINT16
			|| type == ImageFormats::QSampleType::UINT32) {
		sampler = "isampler2D";
	} else if (type == ImageFormats::QSampleType::INT8
			|| type == ImageFormats::QSampleType::INT16
			|| type == ImageFormats::QSampleType::INT32) {
		sampler = "isampler2D";
	} else {
		sampler = "sampler2D";
	}
	return sampler;
}

QString QGLFullCUDARgbaImageItem::createFragmentShader() {
	QString redSampler = samplerType(m_red->sampleType());
	QString greenSampler = samplerType(m_green->sampleType());
	QString blueSampler = samplerType(m_blue->sampleType());
	bool isAlphaDefined = m_alpha!=nullptr;
	QString alphaSampler;
	if (isAlphaDefined) {
		alphaSampler = samplerType(m_alpha->sampleType());
	}

	QStringList out({
			"#version 330 core\n",
			"in vec2 v_textureCoordinates;\n",
			"uniform float f_rangeMinRed = 0.0;\n", // tile min
			"uniform float f_rangeRatioRed = 1.0;\n", // tile min
			"uniform float f_rangeMinGreen = 0.0;\n", // tile min
			"uniform float f_rangeRatioGreen = 1.0;\n", // tile min
			"uniform float f_rangeMinBlue = 0.0;\n", // tile min
			"uniform float f_rangeRatioBlue = 1.0;\n", // tile min
			// tile texture
			"uniform ", redSampler, " f_tileTextureRed;\n",
			"uniform ", greenSampler, " f_tileTextureGreen;\n",
			"uniform ", blueSampler, " f_tileTextureBlue;\n",
	});
	if (isAlphaDefined && m_alphaMode==RgbDataset::OTHER) {
		out << "uniform " << alphaSampler << " f_tileTextureAlpha;\n";
		out << "uniform float f_rangeMinAlpha = 0.0;\n"; // tile min
		out << "uniform float f_rangeRatioAlpha = 1.0;\n"; // tile min
	} else if (m_alphaMode==RgbDataset::TRANSPARENT || m_alphaMode==RgbDataset::OPAQUE) {
		out << "uniform float f_opacityRadius = 0.25;\n";
	}else {// mode NONE
		out << "uniform float f_opacity = 0.25;\n";// tile opacity
	}
	out << "out vec4 f_fragColor;\n"; // shader output color

	out << "void main() {\n";
	out << "float origvalRed = texture(f_tileTextureRed, v_textureCoordinates.st).r;\n";
	out << "float valRed = (origvalRed - f_rangeMinRed) * f_rangeRatioRed;\n";
	out << "if (valRed>1.0)\n valRed=1.0;\n";
	out << "if (valRed<0.0)\n valRed=0.0;\n";
	out << "float origvalGreen = texture(f_tileTextureGreen, v_textureCoordinates.st).r;\n";
	out << "float valGreen = (origvalGreen - f_rangeMinGreen) * f_rangeRatioGreen;\n";
	out << "if (valGreen>1.0)\n valGreen=1.0;\n";
	out << "if (valGreen<0.0)\n valGreen=0.0;\n";
	out << "float origvalBlue = texture(f_tileTextureBlue, v_textureCoordinates.st).r;\n";
	out << "float valBlue  = (origvalBlue - f_rangeMinBlue) * f_rangeRatioBlue;\n";
	out << "if (valBlue>1.0)\n valBlue=1.0;\n";
	out << "if (valBlue<0.0)\n valBlue=0.0;\n";

	if (isAlphaDefined && m_alphaMode==RgbDataset::OTHER) {
		out << "float origvalAlpha = texture(f_tileTextureAlpha, v_textureCoordinates.st).r;\n";
		out << "float valAlpha = (origvalAlpha - f_rangeMinAlpha) * f_rangeRatioAlpha;\n";
		out << "if (valAlpha>1.0)\n valAlpha=1.0;\n";
		out << "if (valAlpha<0.0)\n valAlpha=0.0;\n";
	} else if (m_alphaMode==RgbDataset::TRANSPARENT) {
		out << "float currentRadius = sqrt(pow(valBlue-valRed, 2) + pow(valGreen-valRed, 2) + pow(valGreen-valBlue, 2));\n";
		out << "float valAlpha=1;\n";
		out << "if (f_opacityRadius>currentRadius) {\n";
		out << "valAlpha=0;\n";
		out << "}\n";
	} else if (m_alphaMode==RgbDataset::OPAQUE) {
		out << "float currentRadius = sqrt(pow(valBlue-valRed, 2) + pow(valGreen-valRed, 2) + pow(valGreen-valBlue, 2));\n";
		out << "float valAlpha=1;\n";
		out << "if (f_opacityRadius>currentRadius) {\n";
		out << " valRed=0;\n";
		out << " valGreen=0;\n";
		out << " valBlue=0;\n";
		out << "}\n";
	} else {
		out << "float valAlpha = f_opacity;\n";
	}

	out << "f_fragColor = vec4(valRed,valGreen,valBlue,valAlpha);\n";
	out << "}\n";

	QString outString = out.join("");
	return outString;
}

bool QGLFullCUDARgbaImageItem::loadProgram(QOpenGLShaderProgram *program,
		const QString &vert, const QString &frag) {
    //qDebug() << frag;
	//Shader program initialisation
	if (!program->addShaderFromSourceFile(QOpenGLShader::Vertex, vert))
		return false;
	if (!program->addShaderFromSourceCode(QOpenGLShader::Fragment, frag))
		return false;
	if (!program->link())
		return false;

	return true;
}

void QGLFullCUDARgbaImageItem::initializeShaders() {
	m_program->bind();
	bool done = false;
	do {
		if (!this->loadProgram(m_program, ":shaders/images/lut_vert.glsl",
				createFragmentShader()))
			break;
		done = true;
	} while (0);
	if (!done)
		qDebug() << "Ooops! 1";

	m_program->release();
}

void QGLFullCUDARgbaImageItem::initializeGL() {
	preInitGL();
	initializeShaders();

	// vertex buffer initialisation
	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	// room for 2 triangles of 3 vertices
	m_vertexBuffer.allocate(2 * 3 * sizeof(QVertex2D));

	QVertex2D v0, v1, v2, v3;
	QGLImageItemHelper::computeImageCorner(m_red, v0, v1, v2, v3);

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

	//m_mapper->bindLUTTexture(COLOR_MAP_TEXTURE_UNIT);
	//m_mapper->releaseLUTTexture(COLOR_MAP_TEXTURE_UNIT);

	m_mapperRed->bindTexture(RED_IMAGE_TEXTURE_UNIT);
	m_mapperRed->releaseTexture(RED_IMAGE_TEXTURE_UNIT);

	m_mapperGreen->bindTexture(GREEN_IMAGE_TEXTURE_UNIT);
	m_mapperGreen->releaseTexture(GREEN_IMAGE_TEXTURE_UNIT);

	m_mapperBlue->bindTexture(BLUE_IMAGE_TEXTURE_UNIT);
	m_mapperBlue->releaseTexture(BLUE_IMAGE_TEXTURE_UNIT);

	if (m_alpha!=nullptr) {
		m_mapperAlpha->bindTexture(ALPHA_IMAGE_TEXTURE_UNIT);
		m_mapperAlpha->releaseTexture(ALPHA_IMAGE_TEXTURE_UNIT);
	}

	postInitGL();
	m_initialized = true;
}

void QGLFullCUDARgbaImageItem::setPaletteParameter(QOpenGLShaderProgram *program) {
	QVector2D r = m_red->rangeRatio();
	program->setUniformValue("f_rangeMinRed", r[0]);
	program->setUniformValue("f_rangeRatioRed", r[1]);
	QVector2D g = m_green->rangeRatio();
	program->setUniformValue("f_rangeMinGreen", g[0]);
	program->setUniformValue("f_rangeRatioGreen", g[1]);
	QVector2D b = m_blue->rangeRatio();
	program->setUniformValue("f_rangeMinBlue", b[0]);
	program->setUniformValue("f_rangeRatioBlue", b[1]);
	if (m_alpha!=nullptr) {
		QVector2D a = m_alpha->rangeRatio();
		program->setUniformValue("f_rangeMinAlpha", a[0]);
		program->setUniformValue("f_rangeRatioAlpha", a[1]);
	}
	//program->setUniformValue("f_noHasDataValue", m_image->hasNoDataValue());
	//program->setUniformValue("f_noDataValue", m_image->noDataValue());
}

void QGLFullCUDARgbaImageItem::setOpacity(float opacity) {
	m_opacity = opacity;
	update();
}

void QGLFullCUDARgbaImageItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (!m_initialized)
		initializeGL();


	// program setup
	m_program->bind();
	// a good practice if you have to manage multiple shared context
	// with shared resources: the link is context dependant.
	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}

	// binding the buffer
	m_vertexBuffer.bind();

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
	if (m_alphaMode==RgbDataset::TRANSPARENT || m_alphaMode==RgbDataset::OPAQUE) {
		m_program->setUniformValue("f_opacityRadius", m_radiusAlpha);
	} else if (m_alpha==nullptr) {
		m_program->setUniformValue("f_opacity", m_opacity);
	}
	setPaletteParameter(m_program);

	//m_program->setUniformValue("color_map", COLOR_MAP_TEXTURE_UNIT);
	m_program->setUniformValue("f_tileTextureRed", RED_IMAGE_TEXTURE_UNIT);
	m_program->setUniformValue("f_tileTextureGreen", GREEN_IMAGE_TEXTURE_UNIT);
	m_program->setUniformValue("f_tileTextureBlue", BLUE_IMAGE_TEXTURE_UNIT);
	if( m_alpha!=nullptr) {
		m_program->setUniformValue("f_tileTextureAlpha", ALPHA_IMAGE_TEXTURE_UNIT);
	}

	//m_mapper->bindLUTTexture(COLOR_MAP_TEXTURE_UNIT);
	m_mapperRed->bindTexture(RED_IMAGE_TEXTURE_UNIT);
	m_mapperGreen->bindTexture(GREEN_IMAGE_TEXTURE_UNIT);
	m_mapperBlue->bindTexture(BLUE_IMAGE_TEXTURE_UNIT);
	if (m_alpha!=nullptr) {
		m_mapperAlpha->bindTexture(ALPHA_IMAGE_TEXTURE_UNIT);
	}
	// draw 2 triangles = 6 vertices starting at offset 0 in the buffer
	glDrawArrays(GL_TRIANGLES, 0, 6);
	// release texture
	m_mapperRed->releaseTexture(RED_IMAGE_TEXTURE_UNIT);
	m_mapperGreen->releaseTexture(GREEN_IMAGE_TEXTURE_UNIT);
	m_mapperBlue->releaseTexture(BLUE_IMAGE_TEXTURE_UNIT);
	if (m_alpha!=nullptr) {
		m_mapperAlpha->releaseTexture(ALPHA_IMAGE_TEXTURE_UNIT);
	}
	//m_mapper->releaseLUTTexture(COLOR_MAP_TEXTURE_UNIT);

	m_vertexBuffer.release();
	m_program->release();
}

void QGLFullCUDARgbaImageItem::setRadiusAlpha(float radius) {
	m_radiusAlpha = radius;
	update();
}

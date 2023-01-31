#ifndef QGLGridAxisItem_H
#define QGLGridAxisItem_H


#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

#include <QScopedPointer>
#include <QOpenGLContext>
#include <qopenglextensions.h>

class QOpenGLShaderProgram;
class IGeorefImage;

class QGLGridAxisItem: public QAbstractGLGraphicsItem {
Q_OBJECT
public:
	enum Direction {
		HORIZONTAL, VERTICAL
	};

	QGLGridAxisItem(const QRectF &worldExtent,int expectedPixelsDim, Direction dir, QGraphicsItem *parent=0 );
	~QGLGridAxisItem();
	int AddTicksDefault(const QRectF &exposed, int width,	int height);

	void initializeOpenGLFunctions();

//	void glGetPathSpacingNV(GLenum pathListMode, GLsizei numPaths, GLenum pathNameType, const GLvoid *paths, GLuint pathBase, GLfloat advanceScale, GLfloat kerningScale, GLenum transformType, GLfloat *returnedSpacing);

private:
	void initializeGL();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX, int dpiY)
					override;
	int updateInternalBuffer(const QRectF &exposed,int width,int height);

	void drawText(GLuint glyphBase, const std::string &text);
	void setupShader(const QMatrix4x4 &viewProjectionMatrix,const QMatrix4x4 &matrix, const QVector4D & color);
	void UpdateVertex(const QRectF &exposed);
protected:

	Direction m_dir;
	IGeorefImage *m_image;

	QOpenGLBuffer m_vertexBuffer;
	int m_lineCount;

	bool m_initialized;

	QOpenGLShaderProgram *m_program;
	QVector4D m_color;

	int m_tickSize;
	QScopedPointer<QOpenGLExtension_NV_path_rendering> m_nvPathFuncs;

	std::vector<double> m_tickElements;

	QMatrix4x4 m_transfoMatrix;
	float m_tickRatio;
	int m_numMaxPoints;


	//void (QOPENGLF_APIENTRYP m_getPathSpacingNV)(GLenum , GLsizei , GLenum , const GLvoid *, GLuint , GLfloat , GLfloat , GLenum , GLfloat *);
};

#endif

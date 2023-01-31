#ifndef QGLGridTickItem_H
#define QGLGridTickItem_H

#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

#include <QScopedPointer>
#include <qopenglextensions.h>

class QOpenGLShaderProgram;
class IGeorefImage;

class QGLGridTickItem: public QAbstractGLGraphicsItem {
Q_OBJECT
public:
	QGLGridTickItem(IGeorefImage *image, QGraphicsItem *parent=0);
	~QGLGridTickItem();
private:
	void initializeGL();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX, int dpiY)
					override;

	void addVerticalTick(std::vector<float> *vertexes, float pos, bool major);
	void addHorizontalTick(std::vector<float> *vertexes, float pos, bool major);

	void updateInternalBuffer();

	void drawText(GLuint glyphBase, const std::string &text);
	void setupShader(const QMatrix4x4 &viewProjectionMatrix,const QMatrix4x4 &matrix, const QVector4D & color);
protected:
	IGeorefImage *m_image;

	QOpenGLBuffer m_horizontalVertexBuffer;
	int m_horizotalLineCount;
	QOpenGLBuffer m_verticalVertexBuffer;
	int m_verticalLineCount;
	bool m_initialized;

	QOpenGLShaderProgram *m_program;
	QVector4D m_horizontalColor;
	QVector4D m_verticalColor;

	int m_tickSize;

	QScopedPointer<QOpenGLExtension_NV_path_rendering> m_nvPathFuncs;

	typedef struct {
		float x;
		float y;
		bool major;
		std::string text;
	} TickElement;

	std::vector<TickElement> m_horizontalTickElements;
	std::vector<TickElement> m_verticalTickElements;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */

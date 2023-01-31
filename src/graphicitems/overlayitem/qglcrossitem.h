#ifndef QGLCrossItem_H
#define QGLCrossItem_H

#include <QTransform>
#include <QOpenGLBuffer>
#include "qabstractglgraphicsitem.h"

class QOpenGLFunctions;
class QOpenGLShaderProgram;
class IGeorefImage;
class IPaletteHolder;

class QGLCrossItem: public QAbstractGLGraphicsItem {
Q_OBJECT
public:
	QGLCrossItem(const QRectF &worldExtent, QGraphicsItem *parent = 0);
	~QGLCrossItem();
	void setPosition(double worldX, double worldY);
private:
	void initializeGL();
	void initShaders();
	void setupShader(const QMatrix4x4 &viewProjectionMatrix,
			const QMatrix4x4 &transfo);

	void updateGeometry(const QRectF &exposed);
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX, int dpiY)
					override;

protected:
	bool m_initialized;

	QOpenGLBuffer m_vertexBuffer;
	float m_valuesLine[8];
	QVector4D m_color;

	bool m_needUpdate;

	QOpenGLShaderProgram *m_program;
	QMatrix4x4 m_transfoMatrix;

	int m_posX;
	int m_posY;
};
#endif

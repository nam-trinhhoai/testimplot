#ifndef QGLImageFilledHistogramItem_H
#define QGLImageFilledHistogramItem_H

#include <QTransform>
#include <QOpenGLBuffer>
#include "qabstractglgraphicsitem.h"

class QOpenGLFunctions;
class QOpenGLShaderProgram;
class IGeorefImage;
class IPaletteHolder;

class QGLImageFilledHistogramItem: public QAbstractGLGraphicsItem {
	Q_OBJECT
public:
	QGLImageFilledHistogramItem( IGeorefImage *transfoProvider,IPaletteHolder* holder,QGraphicsItem *parent=0);
	~QGLImageFilledHistogramItem();

	void mouseMove(double worldX, double worldY);

private:
	void initializeGL();
	void initShaders();

	void updateHistogramHorizontalGeometry();
	void updateHistogramVerticalGeometry();

	void setPaletteParameter(QOpenGLShaderProgram* program);
	void setupShader(QOpenGLShaderProgram *program,QVector4D color,const QMatrix4x4 &viewProjectionMatrix);
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix, const QRectF &exposed,int width, int height, int dpiX,int dpiY) override;
protected:
	bool m_initialized;

	//Triangle Strip
	QOpenGLBuffer m_vertexBufferHorizontal;
	std::vector<float> m_valuesHorizontal;
	std::vector<unsigned int > m_indicesHorizontal;

	QOpenGLBuffer m_vertexBufferVertical;
	std::vector<float> m_valuesVertical;
	std::vector<unsigned int > m_indicesVertical;

	//Line
	QOpenGLBuffer m_vertexBufferLineHorizontal;
	float m_valuesLineHorizontal[4];

	QOpenGLBuffer m_vertexBufferLineVertical;
	float m_valuesLineVertical[4];

	QOpenGLShaderProgram* m_program;

	QMatrix4x4 m_transfoMatrix;

	QVector4D m_color;

	IGeorefImage* m_image;
	IPaletteHolder* m_holder;

	int m_posX;
	int m_posY;

	bool m_visible;
	bool m_needUpdate;

	float m_horizontalPixelRatio; // helps to scale the histogram (percentage in term of image pixels dimension)
	float m_verticalPixelRatio; // helps to scale the histogram (percentage in term of image pixels dimension)
};


#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */

#ifndef QGLIsolineItem_H
#define QGLIsolineItem_H

#include "qabstractglgraphicsitem.h"
#include "sliceutils.h"
#include <QGraphicsObject>
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

class CUDAImagePaletteHolder;
class QOpenGLShaderProgram;
class IGeorefImage;

class QGLIsolineItem: public QAbstractGLGraphicsItem {
	Q_OBJECT
public:
	QGLIsolineItem(const IGeorefImage * const transfoProvider,CUDAImagePaletteHolder *isoSurface,int defaultExtractionWndindow,SliceDirection dir,QGraphicsItem *rgtProvider=0);
	~QGLIsolineItem();


	void setColor(QColor c);

public slots:
	void updateSlice(int value);
	void updateWindowSize(unsigned int w);
	void updateRGTPosition();


private:
	void initializeGL();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
					const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;

	void initShaders();

	void updateInternalBuffers();

protected:
	bool m_initialized;
	size_t m_internalBufferSize;

	QOpenGLShaderProgram* m_program;

	QOpenGLBuffer m_vertexBuffer;
	struct cudaGraphicsResource *m_cudaVboResource;

	unsigned int m_integrationWindow;

	CUDAImagePaletteHolder *m_isoSurface;
	unsigned int m_currentPos;

	QVector4D m_color;

	SliceDirection m_dir;
	void * m_cudaBuffer;
	float * m_backingBuffer;

	bool m_needInternalBufferUpdate;

	QMatrix4x4 m_matrix;
};


#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */

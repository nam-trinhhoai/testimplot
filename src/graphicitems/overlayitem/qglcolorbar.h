#ifndef QGLColorScale_H_
#define QGLColorScale_H_

#include "qabstractglgraphicsitem.h"
#include <QVector2D>
#include <QGraphicsObject>
#include <QTransform>

#include <QScopedPointer>
#include <qopenglextensions.h>

#include "lookuptable.h"

class QOpenGLTexture;

class QGLContext;
class QOpenGLBuffer;
class QOpenGLShaderProgram;

class QGLColorBar : public QAbstractGLGraphicsItem
{
	Q_OBJECT
public:
	QGLColorBar(const QRectF &worldExtent,QGraphicsItem *parent=0);
    ~QGLColorBar();

	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix, const QRectF &exposed,int width, int height, int dpiX,int dpiY) override;
public slots:
	void setRange(const QVector2D &range);
	void setOpacity(float value);
	void setLookupTable(const LookupTable & table);
private:
	void initializeGL();
	void drawText(const std::string &text,GLuint glyphBase);
    QOpenGLTexture* generateLUTTexture(bool force);
    void setPaletteParameter(QOpenGLShaderProgram* program);
private:
    QOpenGLBuffer *m_vertexColorTableOverlay;
    QOpenGLShaderProgram* m_patchColorMapProgram;

    QOpenGLTexture* m_colorMapTexture;

    QVector2D m_range;
	float m_opacity;

	LookupTable m_lookupTable;

	bool m_initialized;

	QScopedPointer<QOpenGLExtension_NV_path_rendering> m_nvPathFuncs;

};


#endif /* QTLARGEIMAGEVIEWER_SRC_QTHREADEDGLGRAPHICSSCENE_H_ */

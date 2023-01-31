#ifndef QGLScaleBarItem_H_
#define QGLScaleBarItem_H_

#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>
#include "qopenglext.h"

#include <qopenglextensions.h>

class QOpenGLFunctions;
class QOpenGLTexture;
class QGLContext;
class QOpenGLShaderProgram;
class IGeorefImage;

class QGLScaleBarItem: public QAbstractGLGraphicsItem {
Q_OBJECT
public:
	QGLScaleBarItem(const QRectF &worldExtent, QGraphicsItem *parent=0);
	~QGLScaleBarItem();

	void setMapScale(double value);

private:
	void initializeGL();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;
	void initShaders();

	void setupShader();
	void drawText(GLuint glyphBase, const std::string &text);

	double getRoundIncrement(double startValue);
	double calcMapScale(double widthMap, double widthPage, double mapUnitFactor, int dpi);
	void calcBarScale(int nWidthDC, int nNumTics, double fMapScale, double fBarUnitFactor, int &PixelsPerTic, double &SBUnitsPerTic,int dpi);
protected:
	QOpenGLBuffer m_vertexBuffer;
	QOpenGLBuffer m_ticksVertexBuffer;
	bool m_initialized;
	QOpenGLShaderProgram *m_program;
	QVector4D m_color;

	QScopedPointer<QOpenGLExtension_NV_path_rendering> m_nvPathFuncs;

	float m_tickRatio;
	int m_numMaxPoints;

	static const int PowerRangeMin;
	static const int PowerRangeMax;
	static const int nNiceNumber;
	static const double NiceNumberArray[4];
	static const double metersPerInch;

	double m_mapScale;

	int m_numTicks;

};

#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */

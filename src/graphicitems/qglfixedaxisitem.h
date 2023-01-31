#ifndef QGLFixedAxisItem_H
#define QGLFixedAxisItem_H

#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

#include <QScopedPointer>
#include <qopenglextensions.h>

class QOpenGLShaderProgram;
class IGeorefImage;
class AffineTransformation;

class QGLFixedAxisItem: public QAbstractGLGraphicsItem {
Q_OBJECT
public:
	enum Direction {
		HORIZONTAL, VERTICAL
	};

	/**
	 * WARNING : The image provided in the constructor can be destroyed before the axis item.
	 *
	 * It is not safe to use it outside of the constructor
	 */
	QGLFixedAxisItem(IGeorefImage *image,int expectedPixelsDim, int tickSize,Direction dir, QGraphicsItem *parent=0 );
	~QGLFixedAxisItem();

	void setTextFlip(bool);
	bool textFlip() const;

	QColor color();
	void setColor(const QColor& newColor);
	void resetColor();

	// does a copy of the provided transform
	void setDisplayValueTransform(const AffineTransformation* affineTransform);
	const AffineTransformation* displayValueTransform() const;
private:
	void initializeGL();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX, int dpiY)
					override;

	void addHorizontalTick(std::vector<float> &rVertexes, float pos, bool major);
	void addTick(std::vector<float> &rVertexes, QGLFixedAxisItem::Direction dir,float pos,int index,float maxLen,const QRectF exposed);

	void updateInternalBuffer(const QRectF &exposed, int iWidth, int iHeight);//const QRectF &exposed, int width,	int height);
	void addLine(std::vector<float> &rVertexes,QGLFixedAxisItem::Direction dir,const QRectF exposed);

	void drawText(GLuint glyphBase, const std::string &text);
	void setupShader(const QMatrix4x4 &viewProjectionMatrix,const QMatrix4x4 &matrix, const QVector4D & color);
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

	typedef struct {
		float x;
		float y;
		bool major;
		std::string text;
	} TickElement;

	std::vector<TickElement> m_tickElements;

	bool m_textFlip;
	AffineTransformation* m_displayValueTransform;
};

#endif

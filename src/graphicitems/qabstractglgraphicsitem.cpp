#include "qabstractglgraphicsitem.h"
//#include <QGLWidget>
#include <QOpenGLWidget>
#include <QPaintEngine>
#include <QGraphicsView>
#include <QOpenGLShaderProgram>
#include <iostream>

QAbstractGLGraphicsItem::QAbstractGLGraphicsItem(QGraphicsItem *parent) :
		QGraphicsObject(parent)
{
	setCursor(QCursor(QPixmap(":slicer/icons/crosshair.png")));

}

QAbstractGLGraphicsItem::~QAbstractGLGraphicsItem() {

}
void QAbstractGLGraphicsItem::updateWorldExtent(const QRectF &val) {
	prepareGeometryChange();
	m_worldExtent = val;

}
QOpenGLContext* QAbstractGLGraphicsItem::getContext() {
	QGraphicsView *view = this->scene()->views().first();
	return ((QOpenGLWidget*) (view->viewport()))->context();
//	return ((QGLWidget*) (view->viewport()))->context();
}

QRectF QAbstractGLGraphicsItem::boundingRect() const {
	return m_worldExtent;
}

QMatrix4x4 QAbstractGLGraphicsItem::computeOrthoViewScaleMatrix(int w, int h) {
	QMatrix4x4 combinedMatrix;
	combinedMatrix.scale(2.0 / w, -2.0 / h, 1.0);
	combinedMatrix.translate(-w / 2.0, -h / 2.0);

	return combinedMatrix;
}

bool QAbstractGLGraphicsItem::loadProgram(QOpenGLShaderProgram *program,
		const QString &vert, const QString &frag) {
    qDebug() << frag;
	//Shader program initialisation
	if (!program->addShaderFromSourceFile(QOpenGLShader::Vertex, vert))
		return false;
	if (!program->addShaderFromSourceFile(QOpenGLShader::Fragment, frag))
		return false;
	if (!program->link())
		return false;

	return true;
}

double QAbstractGLGraphicsItem::NiceNum(double x, bool round){
	double f;
	double nf;
	int expv = (int)floor(log10(x));
	f = x / pow(10.0, (double)expv);

	if (round)
		if (f < 1.5)
		{
			nf = 1;
		}
		else if (f < 3)
		{
			nf = 2;
		}
		else if (f < 7)
		{
			nf = 5;
		}
		else
		{
			nf = 10;
		}
	else if (f <= 1)
	{
		nf = 1;
	}
	else if (f <= 2)
	{
		nf = 2;
	}
	else if (f <= 5)
	{
		nf = 5;
	}
	else
	{
		nf = 10;
	}

	return nf * pow(10.0, expv);
}

void QAbstractGLGraphicsItem::paint(QPainter *painter,
		const QStyleOptionGraphicsItem *option, QWidget *widget) {
	if (painter->paintEngine()->type() != QPaintEngine::OpenGL
			&& painter->paintEngine()->type() != QPaintEngine::OpenGL2) {
		qWarning(
				"OpenGLScene: drawBackground needs a QGLWidget to be set as viewport on the graphics view")
				;
		return;
	}
	int logicalDpiX = painter->device()->logicalDpiX();
	int logicalDpiY = painter->device()->logicalDpiY();

	QGraphicsView *view = this->scene()->views().first();
	painter->save();
	painter->beginNativePainting();

	QMatrix4x4 matrix = QMatrix4x4(painter->transform());
	QMatrix4x4 combinedMatrix = computeOrthoViewScaleMatrix(
			painter->device()->width(), painter->device()->height());

	combinedMatrix *= matrix;

	QRectF visibleArea = view->mapToScene(painter->viewport()).boundingRect();
	drawGL(combinedMatrix, mapRectFromScene(visibleArea),
			painter->device()->width(), painter->device()->height(),
			logicalDpiX, logicalDpiY);

	painter->endNativePainting();
	painter->restore();
}


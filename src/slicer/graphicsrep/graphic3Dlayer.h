#ifndef Graphic3DLayer_H
#define Graphic3DLayer_H
#include <QObject>
#include <QVector3D>
#include "qrect3d.h"

class QGraphicsScene;
class QWindow;

namespace Qt3DCore {
class QEntity;
}
namespace Qt3DRender
{
class QCamera;
}

class Graphic3DLayer: public QObject {
Q_OBJECT
public:
	virtual ~Graphic3DLayer();

	virtual void show()=0;
	virtual void hide()=0;

	//the bouding rect is usefull to compute a global BBox for the scene
	virtual QRect3D boundingRect() const = 0;

	virtual void refresh()=0;

	virtual void zScale(float val)=0;

signals:
	void sendInfosCam(QVector3D pos, QVector3D target);

public slots:
	void receiveInfosCam(QVector3D pos, QVector3D target);



protected:
	Graphic3DLayer(QWindow * window,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
protected:
	Qt3DCore::QEntity *m_root;
	Qt3DRender::QCamera * m_camera;
	QWindow * m_window;
};

#endif

#include "graphic3Dlayer.h"

#include <QGraphicsScene>
#include <Qt3DCore/QEntity>


Graphic3DLayer::Graphic3DLayer(QWindow * window,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera):QObject(root)
{
	m_root=root;
	m_camera=camera;
	m_window=window;
}

Graphic3DLayer::~Graphic3DLayer()
{

}

void Graphic3DLayer::receiveInfosCam(QVector3D pos, QVector3D target)
{
	emit sendInfosCam(pos,target);
}


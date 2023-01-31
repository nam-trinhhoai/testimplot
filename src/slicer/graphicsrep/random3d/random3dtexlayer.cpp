#include "random3dtexlayer.h"

#include <iostream>
#include <cmath>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QEntity>
#include <Qt3DExtras/QPlaneMesh>
#include <Qt3DRender/QTechnique>
#include <Qt3DRender/QParameter>
#include <Qt3DInput/QMouseHandler>
#include <Qt3DRender/QPickEvent>
#include <QMouseDevice>
#include <QWindow>
#include <QRenderPass>
#include <QMaterial>
#include <QEffect>
#include <QObjectPicker>
#include <QMouseEvent>
#include <QCamera>
#include <QMatrix4x4>
#include <QDebug>
#include <Qt3DRender/QObjectPicker>
#include <Qt3DRender/QPickingSettings>
#include <Qt3DRender/QPickTriangleEvent>

#include "cudaimagepaletteholder.h"
#include "volumeboundingmesh.h"
#include "seismic3dabstractdataset.h"
#include "LayerSlice.h"
#include "colortabletexture.h"
#include "cudaimagetexture.h"
#include "qt3dhelpers.h"
#include "surfacemesh.h"
#include "layerrgtrep.h"
#include "viewqt3d.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "surfacemeshcacheutils.h"

#include "random3dtexrep.h"
#include "randomtexdataset.h"

Random3dTexLayer::Random3dTexLayer(Random3dTexRep *rep, QWindow *parent,
		Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) :
		Graphic3DLayer(parent, root, camera) {
	m_rep = rep;

}

Random3dTexLayer::~Random3dTexLayer() {


}


RandomTexDataset* Random3dTexLayer::randomData() const {
	return ((RandomTexDataset*) m_rep->data());
}

void Random3dTexLayer::show()
{

	randomData()->parentRandom()->initMaterial(randomData()->texture(),randomData()->range());

}
void Random3dTexLayer::hide()
{

	//randomData()->getRandom3d()->hide();

}

QRect3D Random3dTexLayer::boundingRect() const {

	return QRect3D(0, 0, 0, 1, 1, 1);
}

void Random3dTexLayer::refresh() {

}

void Random3dTexLayer::zScale(float val)
{

}



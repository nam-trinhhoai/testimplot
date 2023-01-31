#include "nurbslayer.h"

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

#include "nurbsrep.h"

NurbsLayer::NurbsLayer(NurbsRep *rep, QWindow *parent,
		Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) :
		Graphic3DLayer(parent, root, camera) {
	m_rep = rep;
/*	m_colorTexture = nullptr;
	m_cudaTexture = nullptr;
	m_cudaSurfaceTexture = nullptr;
	m_surface= nullptr;
	m_grayMaterial = nullptr;

	connect(m_rep->layerSlice()->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), this,
			SLOT(updateLookupTable(const LookupTable &)));
	connect(m_rep->layerSlice()->image(),
			SIGNAL(rangeChanged(const QVector2D &)), this,
			SLOT(rangeChanged()));
	connect(m_rep->layerSlice()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(opacityChanged(float)));

	connect(m_rep->layerSlice()->image(), SIGNAL(dataChanged()), this,
			SLOT(update()));
	connect(m_rep->layerSlice()->isoSurfaceHolder(), SIGNAL(dataChanged()),
			this, SLOT(updateIsoSurface()), Qt::AutoConnection);*/
}

NurbsLayer::~NurbsLayer() {


}


NurbsDataset* NurbsLayer::nurbsData() const {
	return ((NurbsDataset*) m_rep->data());
}

void NurbsLayer::show()
{

	nurbsData()->getNurbs3d()->show();

}
void NurbsLayer::hide()
{

	nurbsData()->getNurbs3d()->hide();

}

QRect3D NurbsLayer::boundingRect() const {

/*	int width = layerSlice()->width();
	int height = layerSlice()->height();
	int depth = layerSlice()->depth();

	QRect3D oriBox = QRect3D(0, 0, 0, width, height, depth);

	// fill list
	double xmin = std::numeric_limits<double>::max();
	double xmax = std::numeric_limits<double>::lowest();
	double ymin = std::numeric_limits<double>::max();
	double ymax = std::numeric_limits<double>::lowest();
	double zmin = std::numeric_limits<double>::max();
	double zmax = std::numeric_limits<double>::lowest();

	QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();

	for (int i=0; i<=width; i+=width) { // xline
		for(int j=0; j<=height; j+=height) { // sample
			for (int k=0; k<=depth; k+= depth) { // inline
				// apply transform
				double iWorld, jWorld, kWorld;
				Seismic3DDataset* dataset = layerSlice()->seismic();
				dataset->ijToXYTransfo()->imageToWorld(i, k, iWorld, kWorld);
				dataset->sampleTransformation()->direct(j, jWorld);

				QVector3D oriPt(iWorld, jWorld, kWorld);
				QVector3D newPoint = transform*oriPt;

				// get min max
				if (xmin>newPoint.x()) {
					xmin = newPoint.x();
				}
				if (xmax<newPoint.x()) {
					xmax = newPoint.x();
				}
				if (ymin>newPoint.y()) {
					ymin = newPoint.y();
				}
				if (ymax<newPoint.y()) {
					ymax = newPoint.y();
				}
				if (zmin>newPoint.z()) {
					zmin = newPoint.z();
				}
				if (zmax<newPoint.z()) {
					zmax = newPoint.z();
				}
			}
		}
	}

	QRect3D worldBox = QRect3D(xmin, ymin, zmin, xmax-xmin, ymax-ymin, zmax-zmin);

	return worldBox;*/

	return QRect3D(0, 0, 0, 1, 1, 1);
}

void NurbsLayer::refresh() {

}

void NurbsLayer::zScale(float val)
{
	/*if (m_surface) {
		m_surface->zScale(val);
	}*/
}



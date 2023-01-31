#include "dataset3Dslicelayer.h"
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
#include <QPhongMaterial>
#include <QEffect>
#include <QObjectPicker>
#include <QTimer>

#include <QCamera>
#include <QDebug>
#include <Qt3DRender/QPickTriangleEvent>


#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "volumeboundingmesh.h"
#include "seismic3dabstractdataset.h"
#include "dataset3Dslicerep.h"
#include "colortabletexture.h"
#include "cudaimagetexture.h"
#include "qt3dhelpers.h"
#include "surfacemesh.h"
#include "viewqt3d.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "surfacemeshcacheutils.h"

Dataset3DSliceLayer::Dataset3DSliceLayer(Dataset3DSliceRep* rep, QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) :
		Graphic3DLayer(parent, root, camera) {
	m_rep = rep;
	m_transform=nullptr;
	m_sliceEntity = nullptr;
	m_colorTexture = nullptr;

	m_cudaTexture = nullptr;

	m_material = nullptr;
	m_opacityParameter = nullptr;
	m_paletteRangeParameter = nullptr;
	m_hoverParameter = nullptr;
	m_line1 = nullptr;
	m_line2 = nullptr;
	m_line3 = nullptr;
	m_line4 = nullptr;



    connect(m_rep->image(), SIGNAL(rangeChanged(const QVector2D &)), this,
                    SLOT(rangeChanged(const QVector2D &)));
    connect(m_rep->image(), SIGNAL(opacityChanged(float)), this,
                    SLOT(opacityChanged(float)));
    connect(m_rep->image(), SIGNAL(lookupTableChanged(const LookupTable &)),
                    this, SLOT(updateLookupTable(const LookupTable &)));


	connect(m_rep->image(), SIGNAL(dataChanged()), this,
			SLOT(update()));
}

Dataset3DSliceLayer::~Dataset3DSliceLayer() {

}

void Dataset3DSliceLayer::opacityChanged(float val) {
	if (m_opacityParameter!=nullptr) {
		m_opacityParameter->setValue(val);
	}
}

void Dataset3DSliceLayer::rangeChanged(const QVector2D &value) {
	if (m_paletteRangeParameter!=nullptr) {
		m_paletteRangeParameter->setValue(
				m_rep->image()->rangeRatio());
	}
}

void Dataset3DSliceLayer::updateTexture(CudaImageTexture *texture,
		CUDAImagePaletteHolder *img) {
	if (texture == nullptr)
		return;

	size_t pointerSize = img->internalPointerSize();
	img->lockPointer();
	texture->setData(
			byteArrayFromRawData((const char*) img->backingPointer(),
					pointerSize));
	img->unlockPointer();
}

void Dataset3DSliceLayer::update() {
	updateTexture(m_cudaTexture, m_rep->image());
	rangeChanged(m_rep->image()->range());

	if (m_transform!=nullptr) {
		int width = dataset()->width();
		int height = dataset()->height();
		int depth = dataset()->depth();
		m_transTr.setToIdentity();
		if (m_rep->direction() == SliceDirection::Inline) {
			QVector3D tr =QVector3D(width*0.5f,height *0.5f, m_rep->currentSliceIJPosition());
			m_transTr.translate(tr);
		}
		else
		{
			QVector3D tr = QVector3D(m_rep->currentSliceIJPosition(),height*0.5f,depth *0.5f);
			m_transTr.translate(tr);
		}
		m_transform->setMatrix(m_scaleTr * m_matrixTr*m_transTr*m_rotationTr);
		//m_transform->setTranslation(m_sliceVector*m_rep->currentSliceIJPosition());
	}
}

void Dataset3DSliceLayer::updateLookupTable(const LookupTable &table) {
        if (m_colorTexture == nullptr)
                return;
        CUDAImagePaletteHolder *img = m_rep->image();
        m_colorTexture->setLookupTable(table);
}


Seismic3DAbstractDataset* Dataset3DSliceLayer::dataset() const {
	return dynamic_cast<Seismic3DAbstractDataset*>(m_rep->data());
}

void Dataset3DSliceLayer::show() {
	int width = dataset()->width();
	int height = dataset()->height();
	int depth = dataset()->depth();

	int width2=width;
	QColor colorline;

	m_sliceEntity = new Qt3DCore::QEntity(m_root);
	//Qt3DExtras::QPlaneMesh* mesh = new Qt3DExtras::QPlaneMesh();
	//SurfaceMesh *mesh = new SurfaceMesh();
	//mesh->setDimensions(QVector2D(1, 1));

	QMatrix4x4 transform;//, rescale;


//	QMatrix4x4 swapAxis;
	/*if (m_rep->direction() == SliceDirection::Inline) {
		rescale.scale(width, 1, height);
		swapAxis = QMatrix4x4(1, 0, 0, 0,
							0, 0, 1, 0,
							0, 1, 0, 0,
							0, 0, 0, 1);
	} else {
		rescale.scale(depth, 1, height);
		QMatrix4x4 swapAxis1(0, 0, 1, 0,
							0, 1, 0, 0,
							1, 0, 0, 0,
							0, 0, 0, 1);
		QMatrix4x4 swapAxis2(0, 1, 0, 0,
							1, 0, 0, 0,
							0, 0, 1, 0,
							0, 0, 0, 1);
		swapAxis = swapAxis2 * swapAxis1;
	}*/

	QMatrix4x4 ijToXYTranform(dataset()->ijToXYTransfo()->imageToWorldTransformation());
	const AffineTransformation* sampleTransform = dataset()->sampleTransformation();

	// swap axis of ijToXYTranform (i,j,k) -> (i,k,j) i:Y, j:Z, k:sample
	const float* tbuf = ijToXYTranform.constData();
	QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
									 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
									 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
									 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);

	double a = sampleTransform->a();
	double b = sampleTransform->b();
	QMatrix4x4 sampleTransfo(1, 0, 0, 0,
							 0, a, 0, b,
							 0, 0, 1, 0,
							 0, 0, 0, 1);
	QMatrix4x4 sceneTransform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
	transform = sceneTransform * ijToXYTranformSwapped * sampleTransfo;// * swapAxis * rescale;
	//mesh->setTransform(transform);


	Qt3DExtras::QPlaneMesh *mesh = new Qt3DExtras::QPlaneMesh();
	if (m_rep->direction() == SliceDirection::Inline) {
		width2 = width;
		colorline = Qt::cyan;
			mesh->setWidth(width);
			mesh->setHeight(height);
			mesh->setMeshResolution(QSize(2, 2));
			mesh->setMirrored(false);

			//m_transform = new Qt3DCore::QTransform();
			QVector3D tr =QVector3D(width*0.5f,height *0.5f, m_rep->currentSliceIJPosition());
			//m_transform->setTranslation(tr);
			m_transTr.setToIdentity();
			m_transTr.translate(tr);

			QQuaternion quat =QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0),-90.0f);
			//m_transform->setRotation(quat);
			m_rotationTr.setToIdentity();
			m_rotationTr.rotate(quat);

	} else {
			width2 = depth;
			colorline = Qt::red;
			mesh->setWidth(depth);
			mesh->setHeight(height);
			mesh->setMeshResolution(QSize(2, 2));
			mesh->setMirrored(false);

			//m_transform = new Qt3DCore::QTransform();
			QVector3D tr = QVector3D(m_rep->currentSliceIJPosition(),height*0.5f,depth *0.5f);
			//m_transform->setTranslation(tr);
			m_transTr.setToIdentity();
			m_transTr.translate(tr);

			QQuaternion quat1 = QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0),-90.0f);
			QQuaternion quat2 = QQuaternion::fromAxisAndAngle(QVector3D(0, 1, 0),-90.0f);

			//m_transform->setRotation(quat2*quat1);
			m_rotationTr.setToIdentity();
			m_rotationTr.rotate(quat2*quat1);
						//	QQuaternion::fromAxisAndAngle(QVector3D(1, 1, 0),-90.0f));
							//QQuaternion::fromAxisAndAngle(QVector3D(0, 1, 0));
	}

	//Create a material
	m_material = new Qt3DRender::QMaterial();

	// Set the effect on the materials
	m_material->setEffect(
			Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/simpleColor.frag",
					"qrc:/shaders/qt3d/simpleColor.vert"));

	// Set different parameters on the materials
	m_colorTexture = new ColorTableTexture();
	CUDAImagePaletteHolder *img = m_rep->image();
	m_cudaTexture = new CudaImageTexture(img->colorFormat(),
			img->sampleType(), img->width(), img->height());
	update();
	updateLookupTable(m_rep->image()->lookupTable());

	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("elementMap"),
					m_cudaTexture));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("colormap"),
					m_colorTexture));

	m_paletteRangeParameter = new Qt3DRender::QParameter(
			QStringLiteral("paletteRange"),
			m_rep->image()->rangeRatio());
	m_material->addParameter(m_paletteRangeParameter);

	m_opacityParameter = new Qt3DRender::QParameter(QStringLiteral("opacity"),
			m_rep->image()->opacity());
	m_material->addParameter(m_opacityParameter);
	m_hoverParameter = new Qt3DRender::QParameter(QStringLiteral("hover"),
			false);
	m_material->addParameter(m_hoverParameter);


	double aX, aY, bX, bY;
	dataset()->ijToXYTransfo()->imageToWorld(0, 0, aX, aY);
	if (m_rep->direction() == SliceDirection::Inline) {
		dataset()->ijToXYTransfo()->imageToWorld(0, 1, bX, bY);
	} else {
		dataset()->ijToXYTransfo()->imageToWorld(1, 0, bX, bY);
	}

//	m_sliceVector = QVector3D((bX-aX), 0, (bY-aY));
	//QVector3D axis(m_sliceVector*m_rep->currentSliceIJPosition());

	update();

	//m_transform = new Qt3DCore::QTransform();
	//m_transform->setScale3D(QVector3D(1, 1, 1));
	//m_transform->setTranslation(axis);
	m_transform = new Qt3DCore::QTransform();


	m_matrixTr =transform ;
	m_transform->setMatrix(m_scaleTr * m_matrixTr*m_transTr*m_rotationTr);

	m_sliceEntity->addComponent(mesh);
	m_sliceEntity->addComponent(m_material);
	m_sliceEntity->addComponent(m_transform);

	m_line1 = Qt3DHelpers::drawLine({ -width2*0.502f, -0, -height*0.502}, { width2*0.502, -0, -height*0.502}, colorline, m_root);
	m_line2 = Qt3DHelpers::drawLine({width2*0.502, -0, -height*0.502 }, { width2*0.502, 0,height*0.5020 }, colorline, m_root);
	m_line3 = Qt3DHelpers::drawLine({ width2*0.502, 0, height *0.502}, { -width2*0.502f,0,height*0.502 }, colorline, m_root);
	m_line4 = Qt3DHelpers::drawLine({ -width2*0.502f,0, height*0.502 }, { -width2*0.502f, -0, -height*0.502 }, colorline, m_root);

/*	Qt3DCore::QTransform* tr1 =  new Qt3DCore::QTransform();
	tr1->setMatrix(m_transform->matrix());
	Qt3DCore::QTransform* tr2 =  new Qt3DCore::QTransform();
	tr2->setMatrix(m_transform->matrix());
	Qt3DCore::QTransform* tr3 =  new Qt3DCore::QTransform();
	tr3->setMatrix(m_transform->matrix());
	Qt3DCore::QTransform* tr4 =  new Qt3DCore::QTransform();
	tr4->setMatrix(m_transform->matrix());
*/
	m_line1->addComponent(m_transform);
	m_line2->addComponent(m_transform);
	m_line3->addComponent(m_transform);
	m_line4->addComponent(m_transform);
	selectPlane(false);




	ViewQt3D* view = dynamic_cast<ViewQt3D*>(m_rep->view());


	connect(this,SIGNAL(sendAnimationCam(int,QVector3D)),view, SLOT(setAnimationCamera(int, QVector3D)),Qt::QueuedConnection);

	  // picker
		Qt3DRender::QObjectPicker* spicker = new Qt3DRender::QObjectPicker();
		Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(spicker);
		pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
		pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
		pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
		pickingSettings->setEnabled(true);
		spicker->setEnabled(true);
		spicker->setDragEnabled(true);

		m_sliceEntity->addComponent(spicker);//m_ghostEntity




		connect(spicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
			m_actifRayon= true;
			});
			connect(spicker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {
			m_actifRayon= false;
			});


		connect(spicker, &Qt3DRender::QObjectPicker::clicked, [&](Qt3DRender::QPickEvent* e) {

		if(m_actifRayon== true)// && e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
		{
			int bouton = e->button();
			auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
			if(p) {
				QVector3D pos = p->worldIntersection();



				//emit sendPositionTarget(newpos,pos);

			emit sendAnimationCam(bouton,pos);
			}
		}
		});

	   // picker
	 /*   picker = new Qt3DRender::QObjectPicker();
	    Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(picker);
	    pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	    pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
	    pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
	    pickingSettings->setEnabled(true);
	    picker->setEnabled(true);
	    picker->setDragEnabled(true);
	    m_sliceEntity->addComponent(picker);

	   connect(picker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {

	    	if(m_movable)
			{


	    		int decalX = e->position().x()-  m_lastPos.x();

	    		movePlane(decalX);
	    		m_lastPos = e->position();
			}

	    });

	    connect(picker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
	    	if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
			{
	    		this->m_movable =true;
	    		selectPlane(this->m_movable);

			}

	      });
	    connect(picker, &Qt3DRender::QObjectPicker::released, [&](Qt3DRender::QPickEvent* e) {
	    	if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
			{
	    		m_movable=false;

	    		selectPlane(false);

			}
	    });
*/


    // picker
 /*   picker = new Qt3DRender::QObjectPicker();
    Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(picker);
    pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
    pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick); //NearestPick);
    pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
    pickingSettings->setEnabled(true);

    picker->setEnabled(true);
    picker->setDragEnabled(true);
    m_sliceEntity->addComponent(picker);


    connect(picker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
    	if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
		{
    		picker->setEnabled(false);

    	//	m_lastPos = e->position();
    		QMatrix4x4 trinverse = (m_scaleTr * m_matrixTr).inverted();

    		QVector3D interIJ  =trinverse * e->worldIntersection();
    		m_lastPosWorld  =interIJ;



    		m_movable =true;


    		qDebug()<<"picker pressed :"<<m_movable;
    		//m_lastPosWorld = e->worldIntersection();
		}

      });
    connect(picker, &Qt3DRender::QObjectPicker::released, [&](Qt3DRender::QPickEvent* e) {
    	if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
		{
    		m_movable=false;
    		//qDebug()<<"picker released :"<<m_movable;
    		selectPlane(false);
    		qDebug()<<"picker released :"<<m_movable;
    		//picker->setEnabled(false);
		}
    });



    connect(picker, &Qt3DRender::QObjectPicker::entered, [&]() {
    	qDebug()<<"picker entered ";
    });

    connect(picker, &Qt3DRender::QObjectPicker::exited, [&]() {
        	qDebug()<<"picker exited ";
        });




   Qt3DCore::QEntity* entity = new Qt3DCore::QEntity(m_camera);
	Qt3DExtras::QPlaneMesh *plane = new Qt3DExtras::QPlaneMesh();
	plane->setWidth(80000);
	plane->setHeight(80000);
	plane->setMeshResolution(QSize(2, 2));
	Qt3DCore::QTransform* transfo = new Qt3DCore::QTransform();

	transfo->setRotationX(90.0f);
	transfo->setTranslation(QVector3D(0.0f,0.0f,-25000.0f));

	Qt3DExtras::QPhongMaterial* mat = new Qt3DExtras::QPhongMaterial(m_root);

	mat->setAmbient(QColor(255, 0, 0, 255));
	mat->setDiffuse(QColor(255, 0, 0, 255));

	entity->addComponent(plane);
	entity->addComponent(mat);
	entity->addComponent(transfo);

	picker2 = new Qt3DRender::QObjectPicker();
	Qt3DRender::QPickingSettings *pickingSettings2 = new Qt3DRender::QPickingSettings(picker2);
	pickingSettings2->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	pickingSettings2->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick); //NearestPick);
	pickingSettings2->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
	pickingSettings2->setEnabled(true);
	picker2->setEnabled(true);

	picker2->setDragEnabled(true);
	//picker2->setHoverEnabled(true);
	entity->addComponent(picker2);

	  connect(picker2, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {

		//  if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
		//  {
			  qDebug()<<" picker2 moved:"<<m_movable;
			  if(m_movable )
			  {

				QMatrix4x4 trinverse = (m_scaleTr * m_matrixTr).inverted();

				QVector3D interIJ  =trinverse * e->worldIntersection();
				QVector3D decal =interIJ -m_lastPosWorld ;

				movePlane(decal);

				//m_lastPosWorld  =interIJ;
				//qDebug()<<" decal:"<<decal;

			 }
		//  }

		});


	   connect(picker2, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
	    	if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
			{


	    		qDebug()<<"picker2 pressed :";
	    		//m_lastPosWorld = e->worldIntersection();
			}

	      });

	  connect(picker2, &Qt3DRender::QObjectPicker::released, [&](Qt3DRender::QPickEvent* e) {
	      	if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
	  		{
	      		qDebug()<<" picker2 released";
	      		picker->setEnabled(true);
	      		selectPlane(false);
	      		 m_movable = false;
	  		}
	  });
*/

}



void Dataset3DSliceLayer::movePlane(QVector3D decal)
{
	//m_transform->setTranslation(decal);
	int width = dataset()->width();
	int height = dataset()->height();
	int depth = dataset()->depth();
	m_transTr.setToIdentity();

//	qDebug()<<" decal:"<<decal;

	if (m_rep->direction() == SliceDirection::Inline) {
		QVector3D tr =QVector3D(width*0.5f,height *0.5f, decal.z());
		m_transTr.translate(tr);
	}
	else
	{
		QVector3D tr = QVector3D(decal.x(),height*0.5f,depth *0.5f);
		m_transTr.translate(tr);
	}
	m_transform->setMatrix(m_scaleTr * m_matrixTr*m_transTr*m_rotationTr);


	/*int current = m_rep->currentSliceWorldPosition();

	int min = (int) m_rep->sliceRangeAndTransfo().first.x();
	int max = (int) m_rep->sliceRangeAndTransfo().first.y();

	if(current+decal >= min && current+decal<= max)
	{

		m_rep->setSliceWorldPosition(current+decal);
	}*/

}


void Dataset3DSliceLayer::movePlane(int decal)
{

	/*m_transTr.setToIdentity();
	if (m_rep->direction() == SliceDirection::Inline) {
		QVector3D tr =QVector3D(width*0.5f,height *0.5f, m_rep->currentSliceIJPosition());
		m_transTr.translate(tr);
	}
	else
	{
		QVector3D tr = QVector3D(m_rep->currentSliceIJPosition(),height*0.5f,depth *0.5f);
		m_transTr.translate(tr);
	}
	m_transform->setMatrix(m_scaleTr * m_matrixTr*m_transTr*m_rotationTr);

*/
	int current = m_rep->currentSliceWorldPosition();

	int min = (int) m_rep->sliceRangeAndTransfo().first.x();
	int max = (int) m_rep->sliceRangeAndTransfo().first.y();

	if(current+decal >= min && current+decal<= max)
	{

		m_rep->setSliceWorldPosition(current+decal);
	}

}

void Dataset3DSliceLayer::selectPlane(bool visible)
{
	m_line1->setEnabled(visible);
	m_line2->setEnabled(visible);
	m_line3->setEnabled(visible);
	m_line4->setEnabled(visible);
}


void Dataset3DSliceLayer::zScale(float val)
{
	m_scaleTr.setToIdentity();
	m_scaleTr.scale(QVector3D(1, val, 1));
	m_transform->setMatrix(m_scaleTr * m_matrixTr*m_transTr*m_rotationTr);
	//m_transform->setScale3D(QVector3D(1, val, 1));
}

void Dataset3DSliceLayer::hide() {
	if(m_sliceEntity != nullptr){
		m_sliceEntity->setParent((Qt3DCore::QEntity*) nullptr);
		m_sliceEntity->deleteLater();
		m_sliceEntity = nullptr;
	}
	if(m_line1 != nullptr){
		m_line1->setParent((Qt3DCore::QEntity*) nullptr);
		m_line1->deleteLater();
		m_line1 = nullptr;
	}
	if(m_line2 != nullptr){
			m_line2->setParent((Qt3DCore::QEntity*) nullptr);
			m_line2->deleteLater();
			m_line2 = nullptr;
		}
	if(m_line3 != nullptr){
			m_line3->setParent((Qt3DCore::QEntity*) nullptr);
			m_line3->deleteLater();
			m_line3 = nullptr;
		}
	if(m_line4 != nullptr){
			m_line4->setParent((Qt3DCore::QEntity*) nullptr);
			m_line4->deleteLater();
			m_line4 = nullptr;
		}
	m_transform = nullptr;
	m_opacityParameter = nullptr;
	m_paletteRangeParameter = nullptr;
	m_hoverParameter = nullptr;
}

QRect3D Dataset3DSliceLayer::boundingRect() const {

	int width = dataset()->width();
	int height = dataset()->height();
	int depth = dataset()->depth();

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
				dataset()->ijToXYTransfo()->imageToWorld(i, k, iWorld, kWorld);
//				iWorld = i*2+1000;
//				kWorld = k*2+1000;
				dataset()->sampleTransformation()->direct(j, jWorld);

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

	return worldBox;
}

void Dataset3DSliceLayer::refresh() {

}


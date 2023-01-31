#include "curvecontroller.h"
#include <QSceneLoader>
#include <QTransform>
#include <QEntity>

#include "meshgeometry.h"
#include "helperqt3d.h"

#include <QVector3D>
#include <Qt3DRender/QEffect>
#include <Qt3DRender/QMaterial>
#include <Qt3DRender/QTechnique>
#include <Qt3DRender/QRenderPass>
#include <Qt3DRender/QShaderProgram>
#include <Qt3DRender/QGraphicsApiFilter>
#include <Qt3DRender/QParameter>

#include <QDiffuseSpecularMaterial>

#include <QPickingSettings>
#include <QPickPointEvent>
#include <QPickLineEvent>

#include <QPhongMaterial>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/vec3.hpp>
#include <glm/gtx/io.hpp>
#include <QMouseEvent>
#include <QMouseHandler>
#include <pointslinesgeometry.h>
#include "nurbsentity.h"
#include <QApplication>
#include <QTimer>
#include <QMouseDevice>


//using namespace Qt3DCore;
using namespace Qt3DRender;


CurveInstantiator::CurveInstantiator(Qt3DCore::QNode *parent)
{
}

void CurveInstantiator::createNew(CurveModel* cm, helperqt3d::IsectPlane plane)
{
    CurveController* cc=new CurveController(this, cm);
    cc->m_isectplane = plane;  // todo: pass plane into constructor instead?

    // When curvemodel is deleted, remove corresponding controller in m_curveControllers and delete it
    connect(cm,  &CurveModel::modeltobeDeleted, this,  &CurveInstantiator::curveDeleted);

    // When curvemodel changes its data, call curvemodelUpdated so geometry can be updated
    connect(cm, &CurveModel::modelUpdated,  cc,  &CurveController::curvemodelUpdated);

    m_curveControllers.push_back(cc);

  //  m_selectedController = cc;
}

CurveModel* CurveInstantiator::getSelectedCurve()
{
     return m_selectedController?m_selectedController->getCurveModel():nullptr;
}

int CurveInstantiator::getPointSelectedIndex()
{
    if (m_selectedController==nullptr) return -1;
    return m_selectedController->m_pointGrabbedId;
}

void CurveInstantiator::unselectSelectedPoint()
{
    m_selectedController->m_pointGrabbedId = -1;
}

// A curve is selected either directly by picking it through Curvecontroller::linePressed event from qt3d or programmatically with this function
void CurveInstantiator::selectCurve(CurveModel* cm)
{
    auto toselect=find_if(m_curveControllers.begin(), m_curveControllers.end(),    // find the curvecontroller that is associated with cm
                            [&] (CurveController* cc) { return cc->getCurveModel() == cm; } );

     if (toselect==m_curveControllers.end()) qDebug("Not found. This should never happen");

     (*toselect)->setSelected(true);
}

// A curve is deselected either directly through Curvecontroller::linePressed event from qt3d or programmatically with this function
void CurveInstantiator::unselectSelectedCurve()
{
    if (!m_selectedController) return;
    m_selectedController->setSelected(false);
    m_selectedController = nullptr;
};


void CurveInstantiator::enablePicking(bool enabled)
{
    for (CurveController* c : m_curveControllers)
    {
        c->enableLinepicking(enabled);
        c->enablePointpicking(false);
        bool isSelected = (m_selectedController == c);
        if (enabled && isSelected)
        {   // When all curves are enabled for picking again, make sure that the one that has previously been selected is still selected (and that points can be moved)
            c->enableLinepicking(false);
            c->enablePointpicking(true);
        }
    }
}

void CurveInstantiator::curveDeleted(CurveModel* cm)
{
    auto todelete=find_if(m_curveControllers.begin(), m_curveControllers.end(),    // find the curvecontroller that is associated with cm
                            [&] (CurveController* cc) { return cc->getCurveModel() == cm; } );

    CurveController* cc = *todelete;
    if (todelete==m_curveControllers.end()) qDebug("Not found. This should never happen");
    m_curveControllers.erase(todelete); //remove curvecontrolller from list

    if (m_selectedController==cc)
        m_selectedController = nullptr;

    delete cc;                   //delete curvecontrolller
}



/*
void CurveInstantiator::setVirtualIsectPlane(QVector3D pointOnModel, QVector3D vector1InPlane, QVector3D vector2InPlane)
{
    m_selectedController->m_isectplane = helperqt3d::IsectPlane{pointOnModel,vector1InPlane,vector2InPlane};
}
*/

// calculates in dxdy the dx and dy in screen coordinates for a plane intersecting
// pointOnModel having in-plane vectors vector1InPlane and vector2InPlane, according to the current view transform
// This can possibly be calculated more easily.
QVector3D CurveInstantiator::unprojectFromScreen(int mouseX, int mouseY) const
{
   QVector3D pointOnModel = m_selectedController->m_isectplane.pointinplane;
   QVector3D vector1InPlane = m_selectedController->m_isectplane.xaxis;
   QVector3D vector2InPlane = m_selectedController->m_isectplane.yaxis;

   QVector3D projected00 = projectToScreen(pointOnModel);
   QVector3D projected01 = projectToScreen(pointOnModel+vector1InPlane);
   QVector3D projected10 = projectToScreen(pointOnModel+vector2InPlane);
   QVector3D vector1InPlaneScreencoords = projected01-projected00;
   QVector3D vector2InPlaneScreencoords = projected10-projected00;

   // (a,b,c) and (d,e,f) are two different vectors on the plane not necessarily orthogonal to each other. Find dx and dy of the plane:
   double a = vector1InPlaneScreencoords.x(), b = vector1InPlaneScreencoords.y(), c = vector1InPlaneScreencoords.z();
   double d = vector2InPlaneScreencoords.x(), e = vector2InPlaneScreencoords.y(), f = vector2InPlaneScreencoords.z();
   // solving a 2x2 system of equations
   double determinant = (a*e)-(b*d);
   double dx = ((c*e)-(b*f)) / determinant;
   double dy = ((a*f)-(c*d)) / determinant;

   QVector2D dxdy  =  QVector2D(dx,dy);
   QVector3D isect =  projectToScreen(pointOnModel);

    ////////////////////


    mouseY =  m_screenheight-mouseY;
    int diffXPix =  mouseX  - isect.x();
    int diffYPix =  mouseY  - isect.y();
    double newz  =  diffXPix*dxdy.x()  + diffYPix*dxdy.y()  +  isect.z();
    QVector3D screenPoswithZ(mouseX,mouseY,newz);

    QRect windowsize(0, 0, m_screenwidth, m_screenheight);
    QMatrix4x4 modelViewMatrix = m_camera->viewMatrix();
    QMatrix4x4 projectionMatrix = m_camera->projectionMatrix();
    return  screenPoswithZ.unproject(modelViewMatrix, projectionMatrix, windowsize);
}


QVector3D CurveInstantiator::projectToScreen(const QVector3D& worldPos) const
{
    QRect windowsize(0, 0, m_screenwidth, m_screenheight);
    QMatrix4x4 modelViewMatrix = m_camera->viewMatrix();
    QMatrix4x4 projectionMatrix = m_camera->projectionMatrix();

    QVector3D pos = worldPos.project(modelViewMatrix, projectionMatrix, windowsize);
    return pos;
}



// ------------------------CURVECONTROLLER---------------------

CurveController::CurveController()
{}

CurveController::~CurveController()
{
   // qDebug("~CurveController");
    delete m_linesEntity;
    delete m_pointsEntity;

    delete m_mouseDevice;
    m_curveInstantiator->removeComponent(m_mouseHandler);
    delete m_mouseHandler;
}

void CurveController::curvemodelUpdated()
{
   // qDebug("CurveController::curvemodelUpdated");
   m_curveGeometry->updateData(m_curveModel->data());
}


CurveController::CurveController(CurveInstantiator* parentEntity, CurveModel* cm) :
    m_curveInstantiator(parentEntity),
    m_curveModel(cm)
{

    // create two renderers/entities, both using the same curvepoint data (curveGeom)

    m_curveGeometry = new PointsLinesGeometry();
    Qt3DCore::QGeometry* curveGeom = m_curveGeometry;

    m_linesEntity =  new Qt3DCore::QEntity(m_curveInstantiator);
    m_materialLine = helperqt3d::makeSimpleMaterial(QColorConstants::Green);
    helperqt3d::makeEntity(m_linesEntity, m_materialLine, curveGeom, true);

    m_pointsEntity =  new Qt3DCore::QEntity(m_curveInstantiator);
    m_materialPoints = helperqt3d::makeSimpleMaterial(QColorConstants::Green);
    helperqt3d::makeEntity(m_pointsEntity, m_materialPoints, curveGeom, false);


    //  ---------------  linePicker  ---------------

   // Qt3DRender::QPickingSettings()
    // Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(m_linePicker);
    // pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::PointPicking);
    // pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
    // pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
    m_linePicker  = new Qt3DRender::QObjectPicker();
    m_linePicker->setPriority(0);
    m_linesEntity->addComponent(m_linePicker);
    m_linePicker->setEnabled(false);
   // m_linePicker->setDragEnabled(true);
   // m_linePicker->setHoverEnabled(true);

    connect(m_linePicker, &Qt3DRender::QObjectPicker::pressed, this, &CurveController::linePressed);


    //  ---------------  pointPicker  ---------------
    m_pointPicker = new Qt3DRender::QObjectPicker();
    m_pointPicker->setPriority(10);
    m_pointsEntity->addComponent(m_pointPicker);
    m_pointPicker->setEnabled(false);   // m_pointPicker->setDragEnabled(true);

    connect(m_pointPicker, &Qt3DRender::QObjectPicker::pressed, this, &CurveController::pointPressed);

    /*
    connect(m_pointPicker, &Qt3DRender::QObjectPicker::moved, [&]
            (Qt3DRender::QPickEvent * e){qDebug() << "pointHover " << ((Qt3DRender::QPickPointEvent*)e)->pointIndex();});
*/
    connect(m_pointPicker, &Qt3DRender::QObjectPicker::entered, [&]
            (){QApplication::setOverrideCursor(Qt::SizeAllCursor); m_curveInstantiator->m_pointHovermode = true; });

    connect(m_pointPicker, &Qt3DRender::QObjectPicker::exited, [&]
            (){QApplication::restoreOverrideCursor(); m_curveInstantiator->m_pointHovermode = false;});
    //See other cursorshapes here : https://doc.qt.io/qt-5/qt.html#CursorShape-enum

    //  ---------------  MOUSE  ---------------
    m_mouseDevice =  new Qt3DInput::QMouseDevice ();
    m_mouseHandler = new Qt3DInput::QMouseHandler();

    // mouse MOVE event
    connect(m_mouseHandler, &Qt3DInput::QMouseHandler::positionChanged, this, &CurveController::mouseMove);

    // mouse RELEASE button event
    connect(m_mouseHandler, &Qt3DInput::QMouseHandler::released,this, &CurveController::mouseButtonRelease);

    m_curveInstantiator->addComponent(m_mouseHandler);

    //  ------------------------------------
}




void CurveController::linePressed(Qt3DRender::QPickEvent* e)
{
    if (m_curveInstantiator->m_pointHovermode) return;

    if (e->buttons() == Qt3DRender::QPickEvent::LeftButton){
  //     Qt3DRender::QPickLineEvent* lineEvent = dynamic_cast<Qt3DRender::QPickLineEvent*>(e);
  //     int lineId = lineEvent->vertex1Index();
  //     qDebug() << "Line pressed " << lineId << "Controller id"  << this;;
        setSelected(true);
    }
}


void CurveController::pointPressed(Qt3DRender::QPickEvent* pickEvent)
{
    pickEvent->setAccepted(true); // has no effect
    Qt3DRender::QPickPointEvent* pointEvent = dynamic_cast<Qt3DRender::QPickPointEvent*>(pickEvent);
    int id = pointEvent->pointIndex();
    m_pointGrabbedId = id;
    //  QVector3D localIsect =  pickEvent->localIntersection();
    //  qDebug() << "screen pos " << pointEvent->position() << " local intersection " << localIsect << " global intersection " <<  pickEvent->worldIntersection() << " Distance " << pickEvent->distance();

    // turn on mouseevents
    m_mouseHandler->setSourceDevice(m_mouseDevice);
}


void CurveController::setSelected(bool setselected)
{
  //  qDebug() << "select" << this << setselected;
    if (setselected)
    {
        if (m_curveInstantiator->m_selectedController)  // some other controller is currently selected so deselect that
            m_curveInstantiator->m_selectedController->setSelected(false);
        m_curveInstantiator->m_selectedController = this;
    }
    else
        m_pointGrabbedId = -1;

    enablePointpicking(setselected);  // turn on  pointpicking for selected curve and vice versa
    enableLinepicking(!setselected);  // turn off linepicking  for selected curve and vice versa
    // linepicking must be turned on for unselected curve so that a click on the line can be detected

    QColor color = setselected?QColorConstants::White:QColorConstants::Green;
    m_materialLine->setAmbient(color);
    m_materialLine->setDiffuse(color);
    m_materialLine->setSpecular(QColor(0,0,0,0));
    m_materialPoints->setAmbient(color);
    m_materialPoints->setDiffuse(color);
    m_materialPoints->setSpecular(QColor(0,0,0,0));

    // let curvemodel emit a selected(true) or (false) event after graphics has been updated
    // so that potential listeners are informed not only when curvemodel is updated, but also when it is (un)selected
    m_curveModel->emitModelSetSelected(setselected);
}



void CurveController::mouseMove(Qt3DInput::QMouseEvent * mouse)
{
    QVector3D mouseIn3D;

    if (!m_isectplane.isUndefined())
        mouseIn3D = m_curveInstantiator->unprojectFromScreen(mouse->x() , mouse->y());
    else
        mouseIn3D = m_curveInstantiator->m_cubePickPos;

    if (m_validPickCoordinate)  // pick coordinate comes from cubepicker and lags one step behind, so dont use first one
        m_curveModel->setPoint(m_pointGrabbedId,mouseIn3D);
    m_validPickCoordinate = true;
}


void CurveController::mouseButtonRelease(Qt3DInput::QMouseEvent * mouse)
{
    //   qDebug() << " mouseButtonRelease";

    // turn off mouseevents and isect with rgt surface
    m_mouseHandler->setSourceDevice(nullptr);

    m_curveModel->emitModelUpdated(true); // To enable redraw of listening geometry when point is released
    //  m_pointGrabbedId = -1;
    //   m_curveInstantiator->m_cubePicker->setEnabled(false);
    //   m_curveInstantiator->m_cubePicker->setDragEnabled(false);
    m_validPickCoordinate = false;
    fixPointPickHack();
}


void CurveController::enablePointpicking(bool enabled)
{
//     qDebug() << " enablePointpicking" << enabled;
    m_pointPicker->setEnabled(enabled);
    m_pointPicker->setDragEnabled(enabled);
    m_pointPicker->setHoverEnabled(enabled);
}

void CurveController::enableLinepicking(bool enabled)
{
 //   qDebug() << " enableLinepicking" << enabled;
    m_linePicker->setEnabled(enabled);
}


// Points become unpickable suddenly without reason. Problem reported to qt3d.
// After experimenting I found out this seems to fix the problem for some reason
void CurveController::fixPointPickHack()
{
    bool wasEnabled    =  m_curveInstantiator->m_cubePicker->isEnabled();
    bool wasDragEnabled = m_curveInstantiator->m_cubePicker->isDragEnabled();

    //The fix: toggling a geometrypicker on and then off seems to update the point geometries bounds also
    m_curveInstantiator->m_cubePicker->setEnabled(true);
    m_curveInstantiator->m_cubePicker->setDragEnabled(true);

    QTimer::singleShot(100, this, [this] () {
        m_curveInstantiator->m_cubePicker->setEnabled(false);
        m_curveInstantiator->m_cubePicker->setDragEnabled(false);
    });
    //////////////////////////////////

    // restore cubepicker back to original state
    QTimer::singleShot(200, this, [this,wasEnabled,wasDragEnabled] () {
        m_curveInstantiator->m_cubePicker->setEnabled(wasEnabled);
        m_curveInstantiator->m_cubePicker->setDragEnabled(wasDragEnabled);
    });
}




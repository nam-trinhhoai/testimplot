
// DEFINES class CurveInstantiator and CurveController

#pragma once

#include <QObject>
#include <QEntity>
#include <QVector2D>
#include <QVector3D>
#include <QObjectPicker>

#include <QCamera>
#include "helperqt3d.h"

class PointsLinesGeometry;
class CurveInstantiator;
class CurveModel;
namespace Qt3DRender
{
    //class QTexture1D;
    //class QComputeCommand;
    //class QRenderStateSet;
    //class QShaderProgram;
    //class QDispatchCompute;
    class QMaterial;
    class QObjectPicker;

    //class QLayer;
    //class QParameter;
    //class QBuffer;
    //class QBufferCapture;
}

namespace Qt3DCore
{
	class QGeometry;
}

namespace Qt3DInput{class QMouseEvent; class QMouseDevice; class QMouseHandler;}
namespace Qt3DExtras{class QDiffuseSpecularMaterial;}

//using namespace Qt3DRender;

// -Takes care of receiving pick events when user clicks on a curve point or a curve segment
// -Renders the curvepoints and segments
// -Each CurveModel is managed by a CurveController. All Curvecontrollers have a pointer to a single CurveInstantiator that manages them
class CurveController  : public QObject
{
    Q_OBJECT
    friend class CurveInstantiator;
public:
    CurveController();
    CurveController(CurveInstantiator* outer, CurveModel* cm);
    ~CurveController();

    CurveModel* getCurveModel(){return m_curveModel;}

private:

    // These privates are accessed by CurveInstantiator
    helperqt3d::IsectPlane  m_isectplane; // Virtual intersection plane for moving curve points on.
    void enablePointpicking(bool enabled);
    void enableLinepicking(bool enabled);
    void curvemodelUpdated();
    void setSelected(bool selected);

    ///////////////////////////////  

    void linePressed(Qt3DRender::QPickEvent* e);
    void pointPressed(Qt3DRender::QPickEvent* e);
    void mouseMove(Qt3DInput::QMouseEvent * mouse);
    void mouseButtonRelease(Qt3DInput::QMouseEvent * mouse);

    void fixPointPickHack();

    CurveInstantiator*    m_curveInstantiator = nullptr;
    CurveModel*               m_curveModel    = nullptr;
    PointsLinesGeometry*      m_curveGeometry = nullptr;
    Qt3DCore::QEntity*        m_linesEntity   = nullptr;
    Qt3DCore::QEntity*        m_pointsEntity  = nullptr;
    Qt3DRender::QObjectPicker*            m_linePicker    = nullptr;
    Qt3DRender::QObjectPicker*            m_pointPicker   = nullptr;
    Qt3DInput::QMouseHandler* m_mouseHandler  = nullptr;
    Qt3DInput::QMouseDevice*  m_mouseDevice   = nullptr;

    int                       m_pointGrabbedId = -1;
    bool                      m_validPickCoordinate=false;
    Qt3DExtras::QDiffuseSpecularMaterial* m_materialLine;
    Qt3DExtras::QDiffuseSpecularMaterial* m_materialPoints;
};


// Manages all the CurveControllers
class CurveInstantiator : public Qt3DCore::QEntity
{
    Q_OBJECT
    friend class CurveController;
public
    slots:  void enablePicking(bool enabled);

public:
    explicit CurveInstantiator(Qt3DCore::QNode *parent = nullptr);

    void setCubePickPos(QVector3D pickPos){if (m_cubePickPos!=pickPos) m_cubePickPos=pickPos;};  QVector3D cubePickPos(){return m_cubePickPos;}
    Q_PROPERTY(QVector3D      cubePickPos  READ cubePickPos      WRITE setCubePickPos        NOTIFY cubePickPosChanged) signals: void cubePickPosChanged(); public:
    Q_PROPERTY(Qt3DRender::QObjectPicker* cubePicker   MEMBER m_cubePicker   NOTIFY cubePickerChanged)   signals: void cubePickerChanged(); public:
    Q_PROPERTY(int screenwidth             MEMBER m_screenwidth  NOTIFY screenwidthChanged)  signals: void screenwidthChanged(); public:
    Q_PROPERTY(int screenheight            MEMBER m_screenheight NOTIFY screenheightChanged) signals: void screenheightChanged(); public:
    Q_PROPERTY(Qt3DRender::QCamera* camera MEMBER m_camera       NOTIFY cameraChanged)       signals: void cameraChanged(); public:

    void        createNew(CurveModel* cm, helperqt3d::IsectPlane plane);
    CurveModel* getSelectedCurve();   
    void        unselectSelectedCurve();
    void        selectCurve(CurveModel* cm);
    int         getPointSelectedIndex();
    void        unselectSelectedPoint();
    QVector3D   unprojectFromScreen(int mouseX, int mouseY) const;


private:
    //  These privates are accessed by CurveController
    Qt3DRender::QObjectPicker*   m_cubePicker = nullptr;  // set from qml
    CurveController* m_selectedController =  nullptr;
    bool             m_pointHovermode = false;
    QVector3D        projectToScreen(const QVector3D& worldPos) const;
    /////////////

    void curveDeleted(CurveModel* cm); // Called from CurveModel::modeltobeDeleted through connect(..)

    // These are set from qml  
    int                  m_screenwidth;
    int                  m_screenheight;
    QVector3D            m_cubePickPos;
    Qt3DRender::QCamera* m_camera;
    //


    std::vector<CurveController*> m_curveControllers;
};


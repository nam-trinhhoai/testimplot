#ifndef CAMERACONTROLLER_H
#define CAMERACONTROLLER_H

#include <Qt3DExtras/QAbstractCameraController>
#include <QQuaternion>
#include <QVector3D>
#include <QSize>
#include <QMatrix4x4>
#include <QRayCaster>
#include <QLayer>
#include "path3d.h"

#include "surfacecollision.h"

namespace Qt3DCore {
class QTransform;
} // Qt3DCore

namespace Qt3DRender {
class QCamera;
} // Qt3DRender

namespace Qt3DLogic {
class QFrameAction;
} // Qt3DLogic

namespace Qt3DInput {
//class QKeyboardDevice;
class QMouseDevice;
class QMouseHandler;
//class QKeyboardHandler;
class QLogicalDevice;
class QAction;
class QActionInput;
class QAxis;
class QAnalogAxisInput;
class QButtonAxisInput;
class QAxisActionHandler;
} // Qt3DInput



class CameraController : public Qt3DCore::QEntity
{
    Q_OBJECT
    Q_PROPERTY(Qt3DRender::QCamera *activeCamera READ activeCamera WRITE setActiveCamera NOTIFY cameraChanged)
    Q_PROPERTY(float panSpeed READ panSpeed WRITE setPanSpeed NOTIFY panSpeedChanged)
    Q_PROPERTY(float zoomSpeed READ zoomSpeed WRITE setZoomSpeed NOTIFY zoomSpeedChanged)
    Q_PROPERTY(float rotationSpeed READ rotationSpeed WRITE setRotationSpeed NOTIFY rotationSpeedChanged)
    Q_PROPERTY(float zoomCameraLimit READ zoomCameraLimit WRITE setZoomCameraLimit NOTIFY zoomCameraLimitChanged)
    Q_PROPERTY(QSize windowSize READ windowSize WRITE setWindowSize NOTIFY windowSizeChanged)
    Q_PROPERTY(float view3DZoomFactor READ view3DZoomFactor NOTIFY view3DZoomFactorChanged)
    Q_PROPERTY(float view2DZoomFactor READ view2DZoomFactor NOTIFY view2DZoomFactorChanged)
    Q_PROPERTY(Mode mode READ mode WRITE setMode NOTIFY modeChanged)
    Q_PROPERTY(QRect viewportRect READ viewportRect WRITE setViewportRect NOTIFY viewportRectChanged)
    Q_PROPERTY(Qt3DCore::QEntity *view3DTarget READ view3DTarget WRITE setView3DTarget NOTIFY view3DTargetChanged)
    Q_PROPERTY(Qt3DRender::QCamera *view3DCamera READ view3DCamera WRITE setView3DCamera NOTIFY view3DCameraChanged)

	Q_PROPERTY(int fps READ fps WRITE setFps NOTIFY fpsChanged)
	Q_PROPERTY(int speedCam READ speedCam WRITE setSpeedCam NOTIFY speedCamChanged)
	Q_PROPERTY(int altitude READ altitude WRITE setAltitude NOTIFY altitudeChanged)

    // The View3DCamera is the camera of the 3D viewport that uses a Perspective/Frustum projection
    // The activeCamera is the camera of the currently selected viewport:
    // - can be the View3DCamera if the currently selected viewport is the 3D viewport
    // - can be an Orthographic camera if the currently selected viewport is a 2D viewport

    // We explicitely need to keep the view3DCamera around because unlike the 2D viewport cameras
    // it needs its viewCenter to track the position of the view3DTarget property

public:

    enum Mode
    {
        Mode3D = 0,
        Mode2D
    };
    Q_ENUM(Mode)

    explicit CameraController(Qt3DCore::QNode *parent = nullptr);
    ~CameraController();

    SurfaceCollision* surfaceCollision() const;
    float view3DZoomFactor() const;
    float view2DZoomFactor() const;
    float panSpeed() const;
    float zoomSpeed() const;
    float moveSpeed() const;
    float rotationSpeed() const;
    float zoomCameraLimit() const;
    QSize windowSize() const;
    Qt3DRender::QCamera *activeCamera() const;
    Mode mode() const;
    QRect viewportRect() const;
    Qt3DCore::QEntity *view3DTarget() const;
    Qt3DRender::QCamera*view3DCamera() const;

    int fps() const;
    int speedCam() const;
    int altitude() const;

    bool modeSurvol() const;

    QMatrix4x4 worldTransformForEntity(Qt3DCore::QEntity *e);

    void setView3DZoomFactor(float f);
    void setView2DZoomFactor(float f);
    void setPanSpeed(float v);
    void setZoomSpeed(float v);
    void setMoveSpeed(float v);
    void setRotationSpeed(float v);
    void setZoomCameraLimit(float v);
    void setWindowSize(const QSize &v);
    void setActiveCamera(Qt3DRender::QCamera *activeCamera);
    void setMode(Mode mode);
    void setViewportRect(const QRect &viewportRect);
    void setView3DTarget(Qt3DCore::QEntity *target);
    void setView3DCamera(Qt3DRender::QCamera *camera);
    void setFps(int v);
    void setFlyCamera();

    void setSpeedCam(int v);
    void setAltitude(int v);

    void setSurfaceCollision(SurfaceCollision* surf);

    void runRayCast();

    void setModeSurvol(bool m);

    void yawPitchMouse(float dt);
    void reinitFocus();

    void setDecalUpDown(float val,float coef =0.0f);

    void setRotSpeed(float value)
    {
    	m_yawpitchSpeed=value;
    }

    void setCoefSpeed(int value)
    {
    	float val= value*0.01f;
    	m_coefSpeed = 0.5f +val;
    }

    void setModePlayPath(bool b)
    {
    	m_indexPlayPath=0;
    	m_modeplayPath =b;

    }
   /* void clearPath()
    {
    	m_paths->ClearAllPoints();

    }
    void savePath()
    {
    	m_paths->SavePath("name");
    }*/

   // void addPositionPath();


    Qt3DRender::QLayer* getLayer()
    {
    	return m_layerNoCast;
    }

    void setTranslationActive(bool b)
    {
    	m_isTranslationActive = b;
    }

    float computeHeight(QVector3D pos);

  /*  void setPathFiles(QString s)
    {
    	m_paths->setPathFiles(s);
    }
*/

public slots:
    void viewEntity(Qt3DCore::QEntity *entity);
    void lookAt(const QVector3D &pos, const QVector3D &at, const QVector3D &up);
    void restoreCustomView();
    void reset3DZoomFactor();
    void onHits(Qt3DRender::QAbstractRayCaster::Hits hits);

signals:
    void cameraChanged();
    void panSpeedChanged();
    void zoomSpeedChanged();
    void moveSpeedChanged();
    void rotationSpeedChanged();
    void zoomCameraLimitChanged();
    void windowSizeChanged();
    void view3DZoomFactorChanged();
    void view2DZoomFactorChanged();
    void modeChanged();
    void cameraTransformChanged();
    void viewportRectChanged();
    void view3DTargetChanged();
    void view3DCameraChanged();

    void fpsChanged();
    void speedCamChanged();
    void altitudeChanged();

    void distanceChanged(float, QVector3D);

    void zoomDistanceChanged(float d);

    void stopModePlay();


private Q_SLOTS:
    void updateCamera3DViewCenter(bool force = false);

private:
    void init();
    void moveCamera(float dt);
    void moveCamXY(float dt);

    void zoomCamera(float amount);
    void rotateCamera(float panAmount, float tiltAmount, float dt);
    void rotateCameraVersion2(float yAxisRotationAmt,
                              float elevationRotationAmt,
                              const QVector3D &initialViewVector);
    void translateCameraXYByPercentOfScreen();
    void trackballRotation();
    void moveCameraFwdBwd(float amount);
    void moveCameraLeftRight(float amount);
    void moveCameraUpDown(float amount);
    void moveCameraPan(float amount);
    void moveCameraTilt(float amount);

    bool isMouseOrbiting() const;
    bool isMouseTranslating() const;
    bool isMouseLateral() const;
    bool isMouseZooming() const;
    bool isRightClickActivate() const;


    void updateCameraTransform(const QQuaternion &rotation);

    Qt3DRender::QCamera *m_activeCamera;
    Qt3DRender::QCamera *m_view3DCamera;

    Qt3DInput::QAction *m_leftMouseButtonAction;
    Qt3DInput::QAction *m_middleMouseButtonAction;
    Qt3DInput::QAction *m_rightMouseButtonAction;

    Qt3DInput::QAxis *m_mouseAxisX;
    Qt3DInput::QAxis *m_mouseAxisY;
  //  Qt3DInput::QAxis *m_keyboardAxisZ;
    Qt3DInput::QAxis *m_mouseWheelAxis;
 /*   Qt3DInput::QAxis *m_keyboardCameraFwdBwd;
    Qt3DInput::QAxis *m_keyboardCameraLeftRight;
    Qt3DInput::QAxis *m_keyboardCameraUpDown;
    Qt3DInput::QAxis *m_keyboardCameraPan;
    Qt3DInput::QAxis *m_keyboardCameraTilt;*/

    Qt3DInput::QMouseDevice *m_mouseDevice;
   // Qt3DInput::QKeyboardDevice *m_keyboardDevice;
    Qt3DInput::QMouseHandler *m_mouseHandler;
   // Qt3DInput::QKeyboardHandler *m_keyboardHandler;
    Qt3DInput::QLogicalDevice *m_logicalDevice;
    Qt3DLogic::QFrameAction *m_frameAction;
    Qt3DCore::QEntity *m_view3DTarget;
    Qt3DCore::QTransform *m_view3DTargetTransform;

    QPoint m_mousePressedPosition;
    QPoint m_mouseCurrentPosition;

    QPoint m_mouseCurrentPositionAll;
    QPoint m_mouseLastCurrentPosition;

    QQuaternion m_cameraRotation;
    QVector3D m_pivotPointAtStart;
    float m_pivotDistance;
    QVector3D m_pivotPoint;
    float m_helperRadius;

    QSize m_windowSize;
    QRect m_viewportRect;
    float m_zoomFactor3D;
    float m_zoomFactor2D;

    //Movement speed control
    float m_panSpeed;
    float m_zoomSpeed;
    float m_rotationSpeed;
    float m_zoomCameraLimit;

    float m_moveSpeed;
    float m_coefSpeed;
    bool m_modeSurvol;

    float m_yawpitchSpeed;

    // holds whether the mouse buttons are set for panning to be active.
    bool MouseCurrentPositionScreen;
    bool m_isTranslationActive;
    bool m_isLateralMove;
    bool m_rightClickActive;
    bool m_wasPressed;


    QMatrix4x4 m_transformMatrix;
    bool m_customCameraTransform;

    Mode m_mode;
    int m_modifiers;
    int m_frameCount;
    float m_elaps;
    int m_fps;
    float m_speedCam;
    float m_altitude;

    float m_lastDistance;
    bool m_reinitDist = true;

   // Qt3DCore::QEntity* m_pointer;
    float m_camera3DInitialViewCenterDistance;

    SurfaceCollision* m_surfaceCollision;
    float m_decalUpDown = 0.0f;

    float m_distanceMin = 5.0f;
    float m_amplitudeDelta = 2.0f;

    float decalY = 0.0f;

    float hauteurcam = 0.0f;
    Qt3DRender::QRayCaster* m_raycast;
    Qt3DRender::QLayer *m_layerNoCast;

    int m_typeMvt=0;
    float m_valueMvt=0.0f;

   // QList<Path3d*> m_pathList3d;
    bool m_modeplayPath = false;
    int m_indexPlayPath=0;

    //Path3d* m_paths = nullptr;

};

#endif // CAMERACONTROLLER_H

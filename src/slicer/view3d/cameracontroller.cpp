#include "cameracontroller.h"
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QCamera>
#include <Qt3DInput/QAction>
#include <Qt3DInput/QActionInput>
#include <Qt3DInput/QAxis>
#include <Qt3DInput/QAnalogAxisInput>
#include <Qt3DInput/QMouseDevice>
#include <Qt3DInput/QKeyboardDevice>
#include <Qt3DInput/QMouseHandler>
#include <Qt3DInput/QKeyboardHandler>
#include <Qt3DInput/QButtonAxisInput>
#include <Qt3DLogic/QFrameAction>
#include <Qt3DInput/QLogicalDevice>
#include <Qt3DCore/private/qentity_p.h>
#include <qmath.h>

#include <iostream>
#include <QPropertyAnimation>

#include "qt3dhelpers.h"

namespace {

float clampInputs(float input1, float input2)
{
    float axisValue = input1 + input2;
    return (axisValue < -1) ? -1 : (axisValue > 1) ? 1 : axisValue;
}

} // anonymous

CameraController::CameraController(Qt3DCore::QNode *parent)
    : Qt3DCore::QEntity(parent)
    , m_activeCamera(nullptr)
    , m_view3DCamera(nullptr)
    , m_leftMouseButtonAction(new Qt3DInput::QAction())
    , m_middleMouseButtonAction(new Qt3DInput::QAction())
    , m_rightMouseButtonAction(new Qt3DInput::QAction())
    , m_mouseAxisX(new Qt3DInput::QAxis())
    , m_mouseAxisY(new Qt3DInput::QAxis())
 /*   , m_keyboardAxisZ(new Qt3DInput::QAxis())
    , m_keyboardCameraFwdBwd(new Qt3DInput::QAxis())
    , m_keyboardCameraLeftRight(new Qt3DInput::QAxis())
    , m_keyboardCameraUpDown(new Qt3DInput::QAxis())
    , m_keyboardCameraPan(new Qt3DInput::QAxis())
    , m_keyboardCameraTilt(new Qt3DInput::QAxis())*/
    , m_mouseWheelAxis(new Qt3DInput::QAxis())
    , m_mouseDevice(new Qt3DInput::QMouseDevice())
 //  , m_keyboardDevice(new Qt3DInput::QKeyboardDevice())
    , m_mouseHandler(new Qt3DInput::QMouseHandler())
 //   , m_keyboardHandler(new Qt3DInput::QKeyboardHandler())
    , m_logicalDevice(new Qt3DInput::QLogicalDevice())
    , m_frameAction(new Qt3DLogic::QFrameAction())
    , m_view3DTarget(nullptr)
    , m_view3DTargetTransform(nullptr)
//	, m_pointer(nullptr)
	, m_surfaceCollision(nullptr)
    , m_windowSize(QSize(1920, 1080))
    , m_zoomFactor3D(2.0f)
    , m_zoomFactor2D(1.0f)
    , m_panSpeed(1.0f)
    , m_zoomSpeed(1.0f)
	, m_moveSpeed(0.0f)
    , m_rotationSpeed(1.0f)
    , m_zoomCameraLimit(1.0f)
    , m_isTranslationActive(false)
	, m_isLateralMove(false)
	, m_rightClickActive(false)
    , m_pivotDistance(1.0f)
    , m_wasPressed(false)
    , m_customCameraTransform(true)
    , m_mode(Mode3D)
	, m_modeSurvol(false)
    , m_modifiers(Qt::NoModifier)
    , m_camera3DInitialViewCenterDistance(0.0f)
	, m_frameCount(0)
	, m_elaps(0.0f)
	, m_fps(0)
	, m_speedCam(0)
	, m_altitude(0)
	, m_lastDistance(0.0)
	, m_coefSpeed(1.0f)
	, m_yawpitchSpeed(2.0f)


{
    init();
}

CameraController::~CameraController()
{
}

float CameraController::view3DZoomFactor() const
{
    return m_zoomFactor3D;
}

float CameraController::view2DZoomFactor() const
{
    return m_zoomFactor2D;
}

void CameraController::init()
{
    //qDebug() << "CameraController::init()";
    //// Actions

    // Left Mouse Button Action
    auto leftMouseButtonInput = new Qt3DInput::QActionInput();
    leftMouseButtonInput->setButtons(QVector<int>() << Qt::LeftButton);
    leftMouseButtonInput->setSourceDevice(m_mouseDevice);
    m_leftMouseButtonAction->addInput(leftMouseButtonInput);

    // Middle Mouse Button Action
    auto middleMouseButtonInput = new Qt3DInput::QActionInput();
    middleMouseButtonInput->setButtons(QVector<int>() << Qt::MiddleButton);
    middleMouseButtonInput->setSourceDevice(m_mouseDevice);
    m_middleMouseButtonAction->addInput(middleMouseButtonInput);

    // Right Mouse Button Action
    auto rightMouseButtonInput = new Qt3DInput::QActionInput();
    rightMouseButtonInput->setButtons(QVector<int>() << Qt::RightButton);
    rightMouseButtonInput->setSourceDevice(m_mouseDevice);
    m_rightMouseButtonAction->addInput(rightMouseButtonInput);



    //// Axes

    // Mouse X
    auto mouseRxInput = new Qt3DInput::QAnalogAxisInput();
    mouseRxInput->setAxis(Qt3DInput::QMouseDevice::X);
    mouseRxInput->setSourceDevice(m_mouseDevice);
    m_mouseAxisX->addInput(mouseRxInput);

    // Mouse Y
    auto mouseRyInput = new Qt3DInput::QAnalogAxisInput();
    mouseRyInput->setAxis(Qt3DInput::QMouseDevice::Y);
    mouseRyInput->setSourceDevice(m_mouseDevice);
    m_mouseAxisY->addInput(mouseRyInput);

    auto mouseWheelXInput = new Qt3DInput::QAnalogAxisInput();
    // Mouse Wheel X
    mouseWheelXInput->setAxis(Qt3DInput::QMouseDevice::WheelX);
    mouseWheelXInput->setSourceDevice(m_mouseDevice);
    m_mouseWheelAxis->addInput(mouseWheelXInput);

    // Mouse Wheel Y
    auto mouseWheelYInput = new Qt3DInput::QAnalogAxisInput();
    mouseWheelYInput->setAxis(Qt3DInput::QMouseDevice::WheelY);
    mouseWheelYInput->setSourceDevice(m_mouseDevice);
    m_mouseWheelAxis->addInput(mouseWheelYInput);

    // Keyboard Z
 /*  auto keyboardZPosInput = new Qt3DInput::QButtonAxisInput();
    keyboardZPosInput->setButtons(QVector<int>() << Qt::Key_PageUp);
    keyboardZPosInput->setScale(1.0f);
    keyboardZPosInput->setSourceDevice(m_keyboardDevice);
    m_keyboardAxisZ->addInput(keyboardZPosInput);

    auto keyboardZNegInput = new Qt3DInput::QButtonAxisInput();
    keyboardZNegInput->setButtons(QVector<int>() << Qt::Key_PageDown);
    keyboardZNegInput->setScale(-1.0f);
    keyboardZNegInput->setSourceDevice(m_keyboardDevice);
    m_keyboardAxisZ->addInput(keyboardZNegInput);

    auto keyboardCameraFwdInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraFwdInput->setButtons(QVector<int>() << Qt::Key_R);
    keyboardCameraFwdInput->setScale(1.0f);
    keyboardCameraFwdInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraFwdBwd->addInput(keyboardCameraFwdInput);

    auto keyboardCameraBackInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraBackInput->setButtons(QVector<int>() << Qt::Key_F);
    keyboardCameraBackInput->setScale(-1.0f);
    keyboardCameraBackInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraFwdBwd->addInput(keyboardCameraBackInput);

    auto keyboardCameraLeftInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraLeftInput->setButtons(QVector<int>() << Qt::Key_D);
    keyboardCameraLeftInput->setScale(1.0f);
    keyboardCameraLeftInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraLeftRight->addInput(keyboardCameraLeftInput);

    auto keyboardCameraRightInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraRightInput->setButtons(QVector<int>() << Qt::Key_G);
    keyboardCameraRightInput->setScale(-1.0f);
    keyboardCameraRightInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraLeftRight->addInput(keyboardCameraRightInput);

    auto keyboardCameraUpInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraUpInput->setButtons(QVector<int>() << Qt::Key_Up);
    keyboardCameraUpInput->setScale(1.0f);
    keyboardCameraUpInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraUpDown->addInput(keyboardCameraUpInput);

    auto keyboardCameraDownInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraDownInput->setButtons(QVector<int>() << Qt::Key_Down);
    keyboardCameraDownInput->setScale(-1.0f);
    keyboardCameraDownInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraUpDown->addInput(keyboardCameraDownInput);

    auto keyboardCameraPanRightInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraPanRightInput->setButtons(QVector<int>() << Qt::Key_P);
    keyboardCameraPanRightInput->setScale(1.0f);
    keyboardCameraPanRightInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraPan->addInput(keyboardCameraPanRightInput);

    auto keyboardCameraPanLeftInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraPanLeftInput->setButtons(QVector<int>() << Qt::Key_I);
    keyboardCameraPanLeftInput->setScale(-1.0f);
    keyboardCameraPanLeftInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraPan->addInput(keyboardCameraPanLeftInput);

    auto keyboardCameraTiltUpInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraTiltUpInput->setButtons(QVector<int>() << Qt::Key_O);
    keyboardCameraTiltUpInput->setScale(1.0f);
    keyboardCameraTiltUpInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraTilt->addInput(keyboardCameraTiltUpInput);

    auto keyboardCameraTiltDownInput = new Qt3DInput::QButtonAxisInput();
    keyboardCameraTiltDownInput->setButtons(QVector<int>() << Qt::Key_L);
    keyboardCameraTiltDownInput->setScale(-1.0f);
    keyboardCameraTiltDownInput->setSourceDevice(m_keyboardDevice);
    m_keyboardCameraTilt->addInput(keyboardCameraTiltDownInput);

*/

    m_layerNoCast = new Qt3DRender::QLayer();
    m_raycast =new Qt3DRender::QRayCaster(this->parentNode());
    m_raycast->setFilterMode(Qt3DRender::QAbstractRayCaster::DiscardAnyMatchingLayers);
    m_raycast->addLayer(m_layerNoCast);
//    m_raycast->setRunMode(Qt3DRender::QAbstractRayCaster::Continuous);

    connect(m_raycast, SIGNAL(hitsChanged(Qt3DRender::QAbstractRayCaster::Hits)),this,SLOT(onHits(Qt3DRender::QAbstractRayCaster::Hits)));
       //// Logical Device

    m_logicalDevice->addAction(m_leftMouseButtonAction);
    m_logicalDevice->addAction(m_middleMouseButtonAction);
    m_logicalDevice->addAction(m_rightMouseButtonAction);
    m_logicalDevice->addAxis(m_mouseAxisX);
    m_logicalDevice->addAxis(m_mouseAxisY);
   /* m_logicalDevice->addAxis(m_keyboardAxisZ);
    m_logicalDevice->addAxis(m_keyboardCameraFwdBwd);
    m_logicalDevice->addAxis(m_keyboardCameraLeftRight);
    m_logicalDevice->addAxis(m_keyboardCameraUpDown);
    m_logicalDevice->addAxis(m_keyboardCameraPan);
    m_logicalDevice->addAxis(m_keyboardCameraTilt);*/
    m_logicalDevice->addAxis(m_mouseWheelAxis);

    //// Mouse handler
    m_mouseHandler->setSourceDevice(m_mouseDevice);

    QObject::connect(m_mouseHandler, &Qt3DInput::QMouseHandler::pressed,
                     [this] (Qt3DInput::QMouseEvent *pressedEvent) {
        if (!isEnabled())
            return;

      //  pressedEvent->setAccepted(true);
        m_wasPressed = true;

        // Record starting point
        m_mousePressedPosition = QPoint(pressedEvent->x(),
                                        pressedEvent->y());
        m_mouseCurrentPosition = m_mousePressedPosition;
        m_cameraRotation = m_activeCamera->transform()->rotation();
        m_pivotPoint = m_activeCamera->viewCenter();
        m_pivotDistance = (m_activeCamera->viewCenter() - m_activeCamera->position()).length();
        m_pivotPointAtStart = m_pivotPoint;

        // translation active if ONLY right button is pressed
        if(pressedEvent->button() == Qt3DInput::QMouseEvent::RightButton)
        {
        	if(m_rightClickActive)
        	{
        		m_isTranslationActive = true;

        		m_mouseLastCurrentPosition = m_mousePressedPosition;
        	}
        	else
        	{
        		m_isLateralMove=true;
        		m_mouseLastCurrentPosition = m_mousePressedPosition;
        	}

        }
        pressedEvent->setAccepted(true);

    });

    QObject::connect(m_mouseHandler, &Qt3DInput::QMouseHandler::wheel,
                     [this] (Qt3DInput::QWheelEvent *wheel) {



    	if(m_rightClickActive)
    		{

    		if(m_moveSpeed < 0.1 && wheel->angleDelta().y()> 0.0 )
    			{
    				decalY = 0.0f;
    				hauteurcam = m_activeCamera->position().y();
    			}
    		// qDebug()<<m_isTranslationActive<<" event wheel :"<<wheel->angleDelta();
    		 m_moveSpeed += wheel->angleDelta().y() *0.05f;
    		 if(m_moveSpeed <0.0f ) m_moveSpeed = 0.0f;
    		 setSpeedCam(m_moveSpeed);
    		}
    	 wheel->setAccepted(true);
        if (!isEnabled())
            return;

    });

    QObject::connect(m_mouseHandler, &Qt3DInput::QMouseHandler::released,
                     [this] (Qt3DInput::QMouseEvent *released) {
        //turn off translation when any button is released.

    	m_isTranslationActive = false;
    	m_isLateralMove =false;
        released->setAccepted(true);
        m_wasPressed = false;

    });

    QObject::connect(m_mouseHandler, &Qt3DInput::QMouseHandler::positionChanged,
                     [this] (Qt3DInput::QMouseEvent *positionChangedEvent) {

    	if(m_isLateralMove || m_isTranslationActive || m_wasPressed)
    	 m_mouseCurrentPosition = QPoint(positionChangedEvent->x(), positionChangedEvent->y());

        if (!isEnabled() || !m_wasPressed){

            return;
    }
        positionChangedEvent->setAccepted(true);

    });

    //// Keyboard handler
  /* m_keyboardHandler->setSourceDevice(m_keyboardDevice);
    m_keyboardHandler->setFocus(true);


    QObject::connect(m_keyboardHandler, &Qt3DInput::QKeyboardHandler::pressed,
                     [this] (Qt3DInput::QKeyEvent *event) {
        m_modifiers = event->modifiers();
        event->setAccepted(true);
    });
    QObject::connect(m_keyboardHandler, &Qt3DInput::QKeyboardHandler::released,
                     [this] (Qt3DInput::QKeyEvent *event) {
        m_modifiers = event->modifiers();
        event->setAccepted(true);
    });*/

    //// FrameAction

    QObject::connect(m_frameAction, &Qt3DLogic::QFrameAction::triggered,
                     this, &CameraController::moveCamera);

    // Disable the logical device when the entity is disabled
    QObject::connect(this, &Qt3DCore::QEntity::enabledChanged,
                     m_logicalDevice, &Qt3DInput::QLogicalDevice::setEnabled);
    QObject::connect(this, &Qt3DCore::QEntity::enabledChanged,
                     m_mouseHandler, &Qt3DInput::QMouseHandler::setEnabled);
    QObject::connect(this, &Qt3DCore::QEntity::enabledChanged,
                     this, [this] (bool enabled){
        if (!enabled)
            m_wasPressed = false;
    });


    //m_paths = new Path3d();


    addComponent(m_frameAction);
    addComponent(m_logicalDevice);
    addComponent(m_mouseHandler);
    addComponent(m_raycast);
//    addComponent(m_keyboardHandler);



}

void CameraController::runRayCast()
{
	m_raycast->trigger(m_activeCamera->position(),m_activeCamera->viewVector().normalized(),40000.0f);
}

void CameraController::onHits(Qt3DRender::QAbstractRayCaster::Hits hits )
{

	if(hits.length()>0)
	{
		emit distanceChanged(hits[0].distance(),hits[0].worldIntersection());
	}
/*	if(m_typeMvt==1)
	{
		if(hits.length()==0)
		{
			m_activeCamera->tilt(-m_valueMvt);
		}
	}
	if(m_typeMvt==2)
	{
		if(hits.length()==0)
		{
			m_activeCamera->pan(-m_valueMvt,QVector3D(0.0f,1.0f,0.0f));
		}
	}*/
	/*for(int i=0;i< hits.length();i++)
	{
		qDebug()<<" hits :"<<hits[i].worldIntersection();
	}*/
}

void CameraController::setSurfaceCollision(SurfaceCollision* surf)
{
	m_surfaceCollision = surf;
}

void CameraController::reinitFocus()
{

//	m_keyboardHandler->setSourceDevice(m_keyboardDevice);

//	m_keyboardHandler->setFocus(true);
}

void CameraController::setFlyCamera()
{
	m_rightClickActive = !m_rightClickActive;
	m_mouseLastCurrentPosition = m_mousePressedPosition;
	//if(m_pointer != nullptr) m_pointer->setEnabled( m_rightClickActive);
	m_moveSpeed=0.0f;
	decalY =0.0f;
	m_isLateralMove = false;
	if(m_rightClickActive ==false)
	{
		m_isTranslationActive = false;
	}
	else{

		QPropertyAnimation* animation = new QPropertyAnimation(m_activeCamera,"upVector");
		animation->setDuration(200);
		animation->setStartValue(m_activeCamera->upVector());
		animation->setEndValue(QVector3D(0.0,-1.0,0.0));
		animation->start();

		if(m_surfaceCollision)
		{
			bool ok;
			m_reinitDist= true;
			m_lastDistance =m_surfaceCollision->distanceSigned(m_activeCamera->position(),&ok);
			if(ok)
			{

				if( m_lastDistance < 10.0f)
				{
					m_activeCamera->translateWorld(QVector3D(0.0, m_lastDistance-10.0f, 0.0));
					m_lastDistance = 10.0f;
					runRayCast();
					setAltitude(m_lastDistance);
				}
			}

		}
	}
}

Qt3DRender::QCamera *CameraController::activeCamera() const
{
    return m_activeCamera;
}

CameraController::Mode CameraController::mode() const
{
    return m_mode;
}

QRect CameraController::viewportRect() const
{
    return m_viewportRect;
}

Qt3DCore::QEntity *CameraController::view3DTarget() const
{
    return m_view3DTarget;
}

Qt3DRender::QCamera *CameraController::view3DCamera() const
{
    return m_view3DCamera;
}

void CameraController::setActiveCamera(Qt3DRender::QCamera *camera)
{
    Qt3DCore::QNodePrivate *d = Qt3DCore::QNodePrivate::get(this);
    if (m_activeCamera != camera) {

        if (m_activeCamera) {
            d->unregisterDestructionHelper(m_activeCamera);
            disconnect(m_activeCamera->transform(), &Qt3DCore::QTransform::matrixChanged, this, &CameraController::cameraTransformChanged);
        }

        if (camera && !camera->parent())
            camera->setParent(this);

        m_activeCamera = camera;

        // Ensures proper bookkeeping
        if (m_activeCamera) {
            d->registerDestructionHelper(m_activeCamera, &CameraController::setActiveCamera, m_activeCamera);
            connect(m_activeCamera->transform(), &Qt3DCore::QTransform::matrixChanged, this, &CameraController::cameraTransformChanged);
        }

     //   m_pointer = Qt3DHelpers::drawLine({ 0, -10, 0 }, { 0, 0, -5000 }, Qt::red, m_activeCamera);
     //   m_pointer->setEnabled(false);
        emit cameraChanged();
        emit cameraTransformChanged();
    }
}

void CameraController::setView3DZoomFactor(float f)
{
    if (m_zoomFactor3D != f) {
        m_zoomFactor3D = f;
        emit view3DZoomFactorChanged();
    }
}

void CameraController::setView2DZoomFactor(float f)
{
    if (m_zoomFactor2D != f) {
        m_zoomFactor2D = f;
        emit view2DZoomFactorChanged();
    }
}

void CameraController::setMode(CameraController::Mode mode)
{
    if (mode == m_mode)
        return;
    m_mode = mode;
    emit modeChanged();
}

void CameraController::setViewportRect(const QRect &viewportRect)
{
    if (viewportRect == m_viewportRect)
        return;
    m_viewportRect = viewportRect;
    emit viewportRectChanged();
}

void CameraController::setView3DTarget(Qt3DCore::QEntity *target)
{
    static QMetaObject::Connection transformChangedConnection;
    if (target == m_view3DTarget)
        return;

    Qt3DCore::QNodePrivate *d = Qt3DCore::QNodePrivate::get(this);
    if (m_view3DTarget) {
        d->unregisterDestructionHelper(m_view3DTarget);
        QObject::disconnect(transformChangedConnection);
    }

    m_view3DTarget = target;
    m_view3DTargetTransform = nullptr;

    // Ensures proper bookkeeping
    if (m_view3DTarget) {
        d->registerDestructionHelper(m_view3DTarget, &CameraController::setView3DTarget, m_view3DTarget);
        const QVector<Qt3DCore::QTransform *> transforms = m_view3DTarget->componentsOfType<Qt3DCore::QTransform>();
        Q_ASSERT(transforms.size() >= 1);
        m_view3DTargetTransform = transforms.first();
        transformChangedConnection = QObject::connect(m_view3DTargetTransform, &Qt3DCore::QTransform::matrixChanged,
                                                      this, [this] () { updateCamera3DViewCenter(); });
    }

    emit view3DTarget();
    updateCamera3DViewCenter(true);
}

void CameraController::setView3DCamera(Qt3DRender::QCamera *camera)
{
    if (m_view3DCamera == camera)
        return;
    m_view3DCamera = camera;
    emit view3DCameraChanged();
}

float CameraController::panSpeed() const
{
    return m_panSpeed;
}

float CameraController::zoomSpeed() const
{
    return m_zoomSpeed;
}

float CameraController::moveSpeed() const
{
    return m_moveSpeed;
}
float CameraController::rotationSpeed() const
{
    return m_rotationSpeed;
}

float CameraController::zoomCameraLimit() const
{
    return m_zoomCameraLimit;
}

int CameraController::fps() const
{
	return m_fps;
}

int CameraController::altitude() const
{
	return m_altitude;
}

int CameraController::speedCam() const
{
	return m_speedCam;
}

SurfaceCollision* CameraController::surfaceCollision() const
{
	return m_surfaceCollision;
}

QSize CameraController::windowSize() const
{
    return m_windowSize;
}

bool CameraController::modeSurvol() const
{
	return m_modeSurvol;
}

void CameraController::setPanSpeed(float v)
{
    if (m_panSpeed != v) {
        m_panSpeed = v;
        emit panSpeedChanged();
    }
}

void CameraController::setFps(int v)
{
	if( m_fps != v)
	{
		m_fps= v;
		emit fpsChanged();
	}
}

void CameraController::setAltitude(int v)
{
	if( m_altitude != v)
	{
		m_altitude= v;
		emit altitudeChanged();
	}
}

void CameraController::setSpeedCam(int v)
{
	if( m_speedCam != v)
	{
		m_speedCam= v;
		emit speedCamChanged();
	}
}

void CameraController::setZoomSpeed(float v)
{
    if (m_zoomSpeed != v) {
        m_zoomSpeed = v;
        emit zoomSpeedChanged();
    }
}

void CameraController::setMoveSpeed(float v)
{
    if (m_moveSpeed != v) {
        m_moveSpeed = v;
        emit moveSpeedChanged();
    }
}

void CameraController::setModeSurvol( bool m)
{
	if(m_modeSurvol != m)
	{
		m_modeSurvol = m;
	}
}

void CameraController::setRotationSpeed(float v)
{
    if (m_rotationSpeed != v) {
        m_rotationSpeed = v;
        emit rotationSpeedChanged();
    }
}

void CameraController::setZoomCameraLimit(float v)
{
    if (m_zoomCameraLimit != v) {
        m_zoomCameraLimit = v;
        emit zoomCameraLimitChanged();
    }
}

void CameraController::setWindowSize(const QSize &v)
{
    if (m_windowSize != v) {
        m_windowSize = v;
        emit windowSizeChanged();
    }
}

void CameraController::viewEntity(Qt3DCore::QEntity *entity)
{
    if (!m_activeCamera)
        return;
    m_activeCamera->lens()->viewEntity(entity->id(), m_activeCamera->id());
}

bool CameraController::isMouseOrbiting() const
{
    return (m_mode == Mode3D) ? m_leftMouseButtonAction->isActive() : false;
}

bool CameraController::isMouseTranslating() const
{

    return m_isTranslationActive;
}

bool CameraController::isMouseLateral() const
{
	return m_isLateralMove;
}

bool CameraController::isRightClickActivate() const
{
    return m_rightClickActive;
}

bool CameraController::isMouseZooming() const
{
    return m_middleMouseButtonAction->isActive() || (m_leftMouseButtonAction->isActive() && m_rightMouseButtonAction->isActive());
}

void CameraController::updateCameraTransform(const QQuaternion & rotation)
{
    // Change camera transform
    QMatrix4x4 viewMatrix;
    viewMatrix.translate(m_pivotPoint);
    viewMatrix.rotate(rotation);
    viewMatrix.translate(QVector3D{0.0f, 0.0f, m_pivotDistance});


    // Extract center, upVector, viewCenter from matrix
    const QVector3D upVector(viewMatrix.row(0)[1],
                             viewMatrix.row(1)[1],
                             viewMatrix.row(2)[1]);
    const QVector3D forward(viewMatrix.row(0)[2],
                            viewMatrix.row(1)[2],
                            viewMatrix.row(2)[2]);

    // Update center/upVector/pos on Camera
    // This is mostly for information and debugging purposes, for actual
    // camera position we set the transformation matrix directly
    m_activeCamera->setViewCenter(m_pivotPoint);
    m_activeCamera->setUpVector(QVector3D::crossProduct(QVector3D::crossProduct(forward,
                                                                          upVector),
                                                  forward));
    m_activeCamera->setPosition(QVector3D(viewMatrix.row(0)[3],
                                    viewMatrix.row(1)[3],
                                    viewMatrix.row(2)[3]));

    // Set transform on camera, usually it is deduced from viewCenter, position, upVector
    // but since we have computed it, use it to avoid cases where above extraction was
    // imprecise
    m_activeCamera->transform()->setMatrix(viewMatrix);
    m_customCameraTransform = true;
}

void CameraController::updateCamera3DViewCenter(bool force)
{
    if (m_view3DCamera == nullptr)
        return;

    // We don't want the activeCamera to move around when we are moving
    // and object that has already been selected
    if (m_activeCamera == m_view3DCamera && !force)
        return;

    QVector3D viewCenter;
    if (m_view3DTarget != nullptr && m_view3DTargetTransform != nullptr) {
        const QMatrix4x4 worldTransform = worldTransformForEntity(m_view3DTarget->parentEntity());
        viewCenter = worldTransform * m_view3DTargetTransform->translation();
    }

    m_view3DCamera->setViewCenter(viewCenter);
}

// move the camera forward and backward
void CameraController::moveCameraFwdBwd(float axisValue)
{
  //  std::cout << "mode survol";
  //  m_modeSurvol = true;
	m_moveSpeed += 1.0*axisValue;
	if(m_moveSpeed< 0.0f) m_moveSpeed =0.0f;
	//qDebug()<<" m_moveSpeed :" <<m_moveSpeed;
	//m_activeCamera->translate(QVector3D(0.0, 0.0, m_moveSpeed*axisValue));
  //  m_view3DCamera->translate(QVector3D(0.0, 0.0, 10.0*axisValue));
}

// move the camera left and right
void CameraController::moveCameraLeftRight(float axisValue)
{
	//m_activeCamera->translate(QVector3D(-m_moveSpeed*axisValue, 0.0, 0.0));
    //m_view3DCamera->translate(QVector3D(-10.0*axisValue, 0.0, 0.0));
}

// move the camera up and down
void CameraController::moveCameraUpDown(float axisValue)
{
	qDebug()<<" OBSOLETE";
	if(m_surfaceCollision)
	{
		bool ok;
		float distance =m_surfaceCollision->distanceSigned(m_activeCamera->position(),&ok);
		setAltitude(distance);
		if(ok && axisValue < 0.0f && distance < 3.0f)
		{
			m_decalUpDown=0.0f;
			return;

		}

	}
	m_decalUpDown = -120.0*axisValue;


	//m_activeCamera->translateWorld(QVector3D(0.0, -120.0*axisValue, 0.0));
    //m_view3DCamera->translate(QVector3D(0.0, -10.0*axisValue, 0.0));
}

float CameraController::computeHeight(QVector3D pos)
{
	if(m_surfaceCollision)
	{
		bool ok;
		float distance =m_surfaceCollision->distanceSigned(pos,&ok);
		if(ok )
		{
			return distance;
		}
	}

	return 0.0f;
}


void CameraController::setDecalUpDown(float val, float coef)
{

	if(m_surfaceCollision)
	{
		bool ok;
		float distance =m_surfaceCollision->distanceSigned(m_activeCamera->position(),&ok);
		setAltitude(distance);
		if(ok && val < 0.0f && distance < 5.0f)
		{
			if(m_moveSpeed > 2.0f)
				m_decalUpDown=0.0;
			else
				if( distance < 1.0f )m_decalUpDown=distance -5.0f;
			return;

		}

	}
	float ajout = 0.0f;
	if(m_moveSpeed > 2.0f) ajout = coef;
	m_decalUpDown = -val+ajout;//-25.0f
	hauteurcam +=m_decalUpDown;


}

// move the camera up and down
void CameraController::moveCameraPan(float axisValue)
{
//	m_activeCamera->pan(10.0*axisValue);
    //m_view3DCamera->pan(10.0*axisValue);
}

// move the camera up and down
void CameraController::moveCameraTilt(float axisValue)
{
//	m_activeCamera->tilt(10.0*axisValue);
   // m_view3DCamera->tilt(10.0*axisValue);
}

// takes normal axis amount from -1 to 1.0
void CameraController::zoomCamera(float axisValue)
{
    if (axisValue < -1.0f || axisValue > 1.0f)
        return;

    // In Ortho mode, we only vary a zoom factor to update the
    // projection left and right coordinates
    // The camera itself doesn't move along the Z axis

    // double the default zoom to make it feel about right
    const float zoomFactorInc = m_zoomSpeed * axisValue;


    // Actually move the camera if we are not using an Ortho Projection
    if (m_activeCamera->projectionType() != Qt3DRender::QCameraLens::OrthographicProjection) {
        const float length = m_activeCamera->viewVector().length();
        float deltaLength = 1.f * zoomFactorInc * length;
        // LOOK UP DETAILS of translate, last param

        m_activeCamera->translate(QVector3D(0.f, 0.f, -deltaLength), Qt3DRender::QCamera::DontTranslateViewCenter);


    }

    if (m_mode == CameraController::Mode3D) {
        if (m_camera3DInitialViewCenterDistance == 0.0f)
            m_camera3DInitialViewCenterDistance = m_activeCamera->viewVector().length();
        m_zoomFactor3D =  m_camera3DInitialViewCenterDistance / m_activeCamera->viewVector().length() * 2.0f;
        float dist = (m_activeCamera->position() - m_activeCamera->viewCenter()).length();
        emit zoomDistanceChanged(dist);
        emit view3DZoomFactorChanged();
    } else {
        m_zoomFactor2D *= 1.0f + (axisValue < 0 ? -0.1f : 0.1f);
        emit view2DZoomFactorChanged();
    }
}

void CameraController::lookAt(const QVector3D &pos, const QVector3D &at, const QVector3D &up)
{
    if (m_customCameraTransform) {
        m_transformMatrix = m_activeCamera->transform()->matrix();
        m_customCameraTransform = false;
    }

    m_activeCamera->setViewCenter(at);
    m_activeCamera->setUpVector(up);
    m_activeCamera->setPosition(pos);
}

void CameraController::translateCameraXYByPercentOfScreen()
{
    // Computes a vector on XY global plane
    // Rotates the vector using the camera rotation, so the vector its on XY camera plane
    // Moves the pivot using the rotated vector
    // Locate the camera on the pivot
    // Rotate the camera using its current rotation
    // Move along the camera Z axis the pivotDistance



	/*m_activeCamera->translate(QVector3D(0.0, 0.0, m_moveSpeed));
    if (m_mousePressedPosition == m_mouseCurrentPosition)
        return;


    QPoint decal = m_mousePressedPosition - m_mouseCurrentPosition;

	m_activeCamera->pan(-0.1f * decal.x());
	m_activeCamera->tilt(0.1f * decal.y());
	m_mousePressedPosition = m_mouseCurrentPosition;*/
    	//	m_mouseLastCurrentPosition.setY(m_mouseCurrentPositionAll.y());

/*    qDebug() << "m_viewportRect" << m_viewportRect.height() << m_viewportRect.width();
    qDebug() << "m_mousePressedPosition.x()" << m_mousePressedPosition.x();

    const QVector3D projectedMousePressedPosition = QVector3D{static_cast<float>(m_mousePressedPosition.x())/m_viewportRect.width() - 0.5f,
            0.5f - static_cast<float>(m_mousePressedPosition.y())/m_viewportRect.height(),
            0.0f};

    const QVector3D projectedMouseCurrentPosition = QVector3D{static_cast<float>(m_mouseCurrentPosition.x())/m_viewportRect.width() - 0.5f,
            0.5f - static_cast<float>(m_mouseCurrentPosition.y())/m_viewportRect.height(),
            0.0f};

    auto worldHeightAtViewCenterForCamera = [] (Qt3DRender::QCamera *camera) {
        if (camera->projectionType() != Qt3DRender::QCameraLens::OrthographicProjection)
            return float(camera->viewVector().length() * tan(camera->fieldOfView() * M_PI / 180.0));
        return camera->top() - camera->bottom();
    };

    auto worldWidthAtViewCenterForCamera = [] (Qt3DRender::QCamera *camera) {
        if (camera->projectionType() != Qt3DRender::QCameraLens::OrthographicProjection)
            return float(camera->viewVector().length() * tan(camera->fieldOfView() * M_PI / 180.0) * camera->aspectRatio());
        return camera->right() - camera->left();
    };

    // Calculate size of view frustum at camera view center.  We'll translate perpendicular to this plane.
    const float worldHeightAtViewCenter = worldHeightAtViewCenterForCamera(m_activeCamera);
    const float worldWidthAtViewCenter = worldWidthAtViewCenterForCamera(m_activeCamera);

    // determine how far the mouse has traveled in plane perp to camera at view center since mouse down
    // take percent mouse has moved in screen and apply to world dimensions from above
    // This is how much we need to translate the camera
    const float percentX = projectedMouseCurrentPosition.x() - projectedMousePressedPosition.x();
    const float percentY = projectedMouseCurrentPosition.y() - projectedMousePressedPosition.y();
    const QVector3D cameraLocalTranslationVec(worldWidthAtViewCenter * percentX, worldHeightAtViewCenter * percentY, 0.0f);

    // move the camera relative to it's starting position to reduce accumated transform error
    // so moving mouse back to initial press position returns camera exactly to initial position.
    QVector3D translatedCameraPos = - cameraLocalTranslationVec;

    QMatrix4x4 mat;
    mat.rotate(m_cameraRotation);
    translatedCameraPos = mat * translatedCameraPos;

    m_pivotPoint = m_pivotPointAtStart + translatedCameraPos;

    updateCameraTransform(m_cameraRotation);*/
}
void CameraController::moveCamXY(float dt)
{


	QPoint decal = m_mouseCurrentPosition - m_mouseLastCurrentPosition;

	float vitessemove = m_activeCamera->position().length()/10.0f;
	bool moveY = true;
	if( abs(decal.x()) > abs(decal.y()))moveY = true;
	else moveY = false;

	if( moveY && m_mouseLastCurrentPosition.x() != m_mouseCurrentPosition.x())
	{
		m_activeCamera->translate(QVector3D(-vitessemove* dt*decal.x(),0.0, 0.0),Qt3DRender::QCamera::TranslateViewCenter);

	}
	if(!moveY &&  m_mouseLastCurrentPosition.y()!= m_mouseCurrentPosition.y())
	{
		m_activeCamera->translate(QVector3D(0.0,vitessemove* dt* decal.y(), 0.0),Qt3DRender::QCamera::TranslateViewCenter);
	}

	//qDebug()<<"moveCamXY   "<<vitessemove;
	m_mouseLastCurrentPosition = m_mouseCurrentPosition;
}


void CameraController::trackballRotation()
{

    // Computes a rotation using a trackball
    // Locates the camera on the pivotPoint
    // Rotates using its previous rotation and the new calculated rotation
    // Move along the camera Z axis the pivotDistance
    if (m_mousePressedPosition == m_mouseCurrentPosition)
        return;

    auto zEvaluator = [] (const QVector3D &projectedPosition) -> float {
        float x2 = std::pow(projectedPosition.x(), 2.0f);
        float y2 = std::pow(projectedPosition.y(), 2.0f);

        float z;
        if (x2 + y2 <= 0.125f) {
            z = std::sqrt(0.25f - (x2 +y2));
        } else {
            z = 0.125f / std::sqrt(x2+y2);
        }
        return z;
    };

    // Project mouse positions to trackballA
    QVector3D projectedMousePressedPosition = QVector3D{static_cast<float>(m_mousePressedPosition.x())/m_windowSize.width()  - 0.5f,
            0.5f - static_cast<float>(m_mousePressedPosition.y())/m_windowSize.height(),
            0.0f};

    QVector3D projectedMouseCurrentPosition = QVector3D{static_cast<float>(m_mouseCurrentPosition.x())/m_windowSize.width()  - 0.5f,
            0.5f - static_cast<float>(m_mouseCurrentPosition.y())/m_windowSize.height(),
            0.0f};

    projectedMouseCurrentPosition.setZ(zEvaluator(projectedMouseCurrentPosition));
    projectedMousePressedPosition.setZ(zEvaluator(projectedMousePressedPosition));
    projectedMouseCurrentPosition.normalize();
    projectedMousePressedPosition.normalize();

    // Find the minimum quaternion that goes from point A to point B
    const QVector3D normal = QVector3D::crossProduct(projectedMousePressedPosition,
                                                     projectedMouseCurrentPosition).normalized();
    const float theta = 100.0f * std::acos(QVector3D::dotProduct(projectedMousePressedPosition,
                                                                 projectedMouseCurrentPosition));
    const QQuaternion rotation = QQuaternion::fromAxisAndAngle(-normal, theta);

    updateCameraTransform(m_cameraRotation * rotation);
}

void CameraController::yawPitchMouse(float dt)
{

	  QVector3D MouseCurrentPositionScreen = QVector3D{static_cast<float>(m_mouseCurrentPosition.x())/m_windowSize.width()  - 0.5f,
		            0.5f - static_cast<float>(m_mouseCurrentPosition.y())/m_windowSize.height(),
		            0.0f};



	//setSpeedCam(dt *m_moveSpeed);


   // QPoint decal = m_mousePressedPosition - m_mouseCurrentPosition;
/*	QVector2D influence(MouseCurrentPositionScreen.x(),MouseCurrentPositionScreen.y());
	if(influence.x() > -0.05 && influence.x() < 0.05) influence.setX(0.0f);
	if(influence.y() > -0.05 && influence.y() < 0.05) influence.setY(0.0f);

	m_activeCamera->pan(1.1f * influence.x());
	m_activeCamera->tilt(1.1f * influence.y());
	m_mousePressedPosition = m_mouseCurrentPosition;*/


	QPoint decal = m_mouseCurrentPosition - m_mouseLastCurrentPosition;

	bool pivotY = true;
	if( abs(decal.x()) > abs(decal.y()))pivotY = true;
	else pivotY = false;

	if( pivotY && m_mouseLastCurrentPosition.x() != m_mouseCurrentPosition.x())
	{
	//	m_valueMvt = -dt *6.01f * decal.x();
	//	m_typeMvt=2;
		m_activeCamera->pan(-dt *m_yawpitchSpeed * decal.x(),QVector3D(0.0f,1.0f,0.0f));
	//	m_raycast->trigger(m_activeCamera->position(),m_activeCamera->viewVector().normalized(),10000.0f);

		//m_mouseLastCurrentPosition.setX( m_mouseCurrentPositionAll.x());

	}
	if(!pivotY &&  m_mouseLastCurrentPosition.y()!= m_mouseCurrentPosition.y())
	{
	//	m_valueMvt = -dt *6.01f * decal.y();
	//	m_typeMvt=1;

		if(m_activeCamera->viewVector().normalized().y() < 0.98f || decal.y() < 0.0f)
				m_activeCamera->tilt(-dt *m_yawpitchSpeed * decal.y());
	//	m_raycast->trigger(m_activeCamera->position(),m_activeCamera->viewVector().normalized(),10000.0f);

	//	m_mouseLastCurrentPosition.setY(m_mouseCurrentPositionAll.y());
	}





	m_mouseLastCurrentPosition = m_mouseCurrentPosition;
}

/*
void CameraController::moveCamera(float dt)
{
	 if (m_activeCamera == nullptr || !isEnabled())
	        return;

	 //compute fps
	 ++m_frameCount;
	m_elaps += dt*1000;
	if (m_elaps >= 1000)
	{
		 setFps((int)( 1.0f /dt));
	}

	//mode helico
	if (isRightClickActivate())
	{
		float decal = 0.0f;
		float repos = 0.0f;
		if(m_moveSpeed> 0.0f) //si je me deplace
		{
			if(m_surfaceCollision )	// si un terrain existe
			{
				bool ok=true;
				float distance =m_surfaceCollision->distanceSigned(m_activeCamera->position(),&ok);
				setAltitude(distance);
				if(ok) //collision ok
				{

					if( distance < 5.0f) //si je suis trop proche du sol
					{
						repos = 5.0f - distance;
						m_lastDistance -=m_decalUpDown-repos;

					}
					else	//sinon je suis au dessus du terrain
					{
						if(m_reinitDist)
						{
							m_lastDistance = distance;
							m_reinitDist =false;
						}

						if( distance <m_lastDistance + m_amplitudeDelta && distance >m_lastDistance - m_amplitudeDelta)
						{

						}
						else
						{
							if(distance > m_lastDistance + m_amplitudeDelta)//je suis trop haut
								decal= distance - m_lastDistance- m_amplitudeDelta;
							if(distance < m_lastDistance - m_amplitudeDelta)//je suis trop bas
								decal= distance - m_lastDistance+ m_amplitudeDelta;
						}
					}
				}
				else // a l'exterieur du terrain
				{
					m_reinitDist =true;
				}
			}
		}

		m_lastDistance -=m_decalUpDown;



		float moveY = decal + m_decalUpDown-repos;
		//deplacement camera (m_moveSpeed ( forward), m_decalUpDown + decal ( up) )
		QVector3D dir1 = QVector3D(0.0,0.0,- dt *m_moveSpeed);
		QVector3D dirCam =  m_activeCamera->transform()->rotation().inverted().rotatedVector(dir1);
		QVector3D dirT = dirCam + QVector3D(0.0f,moveY,0.0f);
		m_activeCamera->translateWorld(dirT);

		//rotation pivot camera with mouse move
		if(isMouseTranslating())
		{
			yawPitchMouse(dt);
			return;
		}
	}
	//rotate  around centerView (orbital)
	if (isMouseOrbiting())
	{
		trackballRotation();
		return;
	}

	 //zoom wheel mouse
	const float mouseWheelValue = (m_modifiers & Qt::ControlModifier) ? 0.0f : m_mouseWheelAxis->value();
	if (!isRightClickActivate())
	{
		if (!qFuzzyCompare( mouseWheelValue, 0.0f)) {
			zoomCamera( mouseWheelValue * dt);
		}
	}
}*/
/*
void CameraController::addPositionPath()
{
	//m_paths->AddPoints(m_activeCamera->position(),m_activeCamera->viewCenter(),2.0f);
	//m_pathList3d.append(new Path3d(tmp,m_activeCamera->viewCenter()));
}*/

void CameraController::moveCamera(float dt)
{

    // Always maintain viewCenter to what was set originally
    if (m_activeCamera == nullptr || !isEnabled())
        return;

    ++m_frameCount;
    m_elaps += dt*1000;
    if (m_elaps >= 1000)
     {
    	  setFps((int)( 1.0f /dt));// / (double)m_frameCount;
     }

	if (isRightClickActivate())
	{
		runRayCast();
		//m_raycast->trigger(m_activeCamera->position(),m_activeCamera->viewVector().normalized(),40000.0f);
		if(m_moveSpeed> 0.0f)
		{
			if(m_surfaceCollision)
			{
				bool ok=true;
				float distance =m_surfaceCollision->distanceSigned(m_activeCamera->position(),&ok);

				if(ok && distance >= 5.0f )
				{
					setAltitude(distance);

					if(m_reinitDist)
					{
						m_lastDistance = distance;
						m_reinitDist =false;
					}

					float decal = qMax(10.0f-distance+m_decalUpDown ,-hauteurcam+ m_activeCamera->position().y() );
					QVector3D dir1 = QVector3D(0.0,0.0,- dt *m_moveSpeed*m_coefSpeed);
					QVector3D dirCam =  m_activeCamera->transform()->rotation().inverted().rotatedVector(dir1);
					QVector3D dirT = dirCam + QVector3D(0.0f,-decal,0.0f);//distance-m_lastDistance+
					m_activeCamera->translateWorld(dirT);

					if(m_decalUpDown != 0)
					{
						m_lastDistance -=m_decalUpDown;
					}
				}
				else
				{
					if(ok && distance < 5.0f)
					{
						float repos = 5.0f - distance;
						m_lastDistance -=m_decalUpDown-repos;

						QVector3D dir1 = QVector3D(0.0,0.0,- dt *m_moveSpeed*m_coefSpeed);
						QVector3D dirCam =  m_activeCamera->transform()->rotation().inverted().rotatedVector(dir1);
						QVector3D dirT = dirCam + QVector3D(0.0f,m_decalUpDown-repos,0.0f);
						m_activeCamera->translateWorld(dirT);
						//m_activeCamera->translateWorld(QVector3D(0.0, m_decalUpDown-repos, 0.0));
					}
					if(!ok)
					{
						m_reinitDist =true;
						QVector3D dir1 = QVector3D(0.0,0.0,- dt *m_moveSpeed*m_coefSpeed);
						QVector3D dirCam =  m_activeCamera->transform()->rotation().inverted().rotatedVector(dir1);
						QVector3D dirT = dirCam + QVector3D(0.0f,m_decalUpDown,0.0f);
						m_activeCamera->translateWorld(dirT);
						//m_lastDistance -=m_decalUpDown;
					}
					if(m_decalUpDown != 0)
					{
						m_lastDistance -=m_decalUpDown;
					}
				}
			}
			else
			{
				QVector3D dir1 = QVector3D(0.0,0.0,- dt *m_moveSpeed*m_coefSpeed);
				QVector3D dirCam =  m_activeCamera->transform()->rotation().inverted().rotatedVector(dir1);

				QVector3D dirT = dirCam + QVector3D(0.0f,m_decalUpDown,0.0f);
				m_activeCamera->translateWorld(dirT);
				//setAltitude(-m_activeCamera->position().y());
			}
		}
		else{
			if(m_decalUpDown != 0.0)
			{

				m_lastDistance -=m_decalUpDown;
				m_activeCamera->translateWorld(QVector3D(0.0, m_decalUpDown, 0.0),Qt3DRender::QCamera::DontTranslateViewCenter);
			}
		}
		if(isMouseTranslating())
		{
			yawPitchMouse(dt) ;//translateCameraXYByPercentOfScreen();
			return;
		}
	}

	else if (isMouseOrbiting()) { // checking if the left mouse button is pressed, maybe change this to get the spaceship behavior
	        trackballRotation();
	        return;
	    }
	else if(isMouseLateral())
	{
		moveCamXY(dt);
		return;
	}
	else if(m_decalUpDown != 0.0)
	{

			m_lastDistance -=m_decalUpDown;
			m_activeCamera->translateWorld(QVector3D(0.0, m_decalUpDown, 0.0),Qt3DRender::QCamera::DontTranslateViewCenter);

	}



	//}
    // zoom via keyboard or mouse wheel
    const float keyboardMotionScale = 0.25f * dt;


    const float mouseWheelValue = (m_modifiers & Qt::ControlModifier) ? 0.0f : m_mouseWheelAxis->value();

    if (!isRightClickActivate())
    {
		if (!qFuzzyCompare( mouseWheelValue, 0.0f)) {
			// TO DO: Check start values (m_cameraRotation, m_pivotPoint, m_pivotPointDistance) when using keyboard
			zoomCamera( clampInputs(keyboardMotionScale, mouseWheelValue * dt));
		}
    }



}

void CameraController::restoreCustomView()
{
    if (!m_customCameraTransform) {
        m_customCameraTransform = true;
        m_activeCamera->transform()->setMatrix(m_transformMatrix);
    }
}

void CameraController::reset3DZoomFactor()
{
    setView3DZoomFactor(1.0f);
}

QMatrix4x4 CameraController::worldTransformForEntity(Qt3DCore::QEntity *e)
{
    QMatrix4x4 transformMatrix;

    while (e != nullptr) {
        const QVector<Qt3DCore::QTransform *> transforms = e->componentsOfType<Qt3DCore::QTransform>();
        if (transforms.size() > 0) {
            const Qt3DCore::QTransform *transform = transforms.first();
            transformMatrix = transform->matrix() * transformMatrix;
        }
        e = e->parentEntity();
    }

    return transformMatrix;
}

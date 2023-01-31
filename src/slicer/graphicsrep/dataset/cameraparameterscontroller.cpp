#include "cameraparameterscontroller.h"
#include "idata.h"
#include "datasetrep.h"

CameraParametersController::CameraParametersController(DatasetRep *rep,ViewQt3D* view3d, QObject *parent) :
	DataControler(parent) {
	m_rep = rep;
	m_view3d = view3d;

	m_position = view3d->positionCam();
	m_target =QVector3D(0.0f,5000.0f,1000000.0f);// view3d->targetCam();
	m_distanceTarget = 10000.0f;

	//m_position = rep->currentSliceWorldPosition();

	connect(view3d, SIGNAL(positionCamChangedSignal(QVector3D)), this,SLOT(setPositionFromRep(QVector3D)));
	connect(view3d, SIGNAL(viewCenterCamChangedSignal(QVector3D)), this,SLOT(setTargetFromRep(QVector3D)));

	connect(view3d, SIGNAL(showHelico2D(bool)), this ,SLOT(showHelico(bool)));

	connect(view3d, SIGNAL(distanceTargetChangedSignal(float)), this ,SLOT(setDistanceTargetFromRep(float)));
}

CameraParametersController::~CameraParametersController() {

}



void CameraParametersController::requestPosChanged(QVector3D val)
{
	m_position = val;
	QVector3D position = m_view3d->sceneTransform()   * val;

	m_view3d->setPositionCam(position);


//	m_rep->setSliceWorldPosition(val);
}

void CameraParametersController::requestPosXChanged(float x)
{
	qDebug()<< " x changed "<<x;
	//QVector3D position = m_view3d->sceneTransform()   * val;
//	m_view3d->positionCameraChanged(position);

}

void CameraParametersController::requestPosYChanged(float y)
{
	qDebug()<< " y changed "<<y;
//	QVector3D position = m_view3d->sceneTransform()   * val;
	//m_view3d->positionCameraChanged(position);

}

QUuid CameraParametersController::dataID()const {
	return m_rep->data()->dataID();
}

QVector3D CameraParametersController::position() const {
	return m_position;
}

QVector3D CameraParametersController::target() const {
	return m_target;
}

float CameraParametersController::distanceTarget() const {
	return m_distanceTarget;
}

void CameraParametersController::requestTargetChanged(QVector3D  val)
{
	m_target = val;
	QVector3D target = m_view3d->sceneTransform()   * val;

	m_view3d->setViewCenterCam(target);

	m_view3d->setUpVectorCam(QVector3D(0.0f,-1.0f,0.0f));
}

void CameraParametersController::setTargetFromRep(QVector3D target)
{

	if(!m_view3d->getAnimRunning())
	{


		m_target =  m_view3d->sceneTransformInverse()   *target;
		emit targetChanged(m_target);

		m_distanceTarget=  (m_position-m_target).length(),
		emit distanceTargetChanged(m_distanceTarget);
	}
}

void CameraParametersController::setDistanceTargetFromRep(float d)
{
	if(!m_view3d->getAnimRunning())
	{

		m_distanceTarget =d;
		emit distanceTargetChanged(m_distanceTarget);
	}
}
/*
void CameraParametersController::setPosition(QVector3D pos) {

	m_position = pos;
	emit positionToRepChanged(pos);
}*/

void CameraParametersController::setPositionFromRep(QVector3D pos)
{

	if(!m_view3d->getAnimRunning())
	{

		m_position = m_view3d->sceneTransformInverse()   * pos;
		emit posChanged(m_position);
	}
}


void CameraParametersController::setRefreshPosition(QVector3D pos)
{
	m_position = pos;
}

bool CameraParametersController::helicoVisible( ) const
{
	return m_helicoVisible;
}

void CameraParametersController::showHelico(bool b)
{
	m_helicoVisible = b;
	emit helicoShowed(b);
}

void CameraParametersController::showLineVert(bool b)
{
	m_view3d->showLineVert(b);
}

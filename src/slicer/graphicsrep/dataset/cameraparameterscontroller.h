#ifndef CameraParametersController_h
#define CameraParametersController_h

#include <QObject>
#include <QVector3D>

#include "datacontroler.h"
#include "viewqt3d.h"

class DatasetRep;
class CameraParametersController : public DataControler{
	  Q_OBJECT
public:
	  CameraParametersController(DatasetRep *rep,ViewQt3D* view3d, QObject *parent);
	virtual ~CameraParametersController();

	QVector3D position() const;
	QVector3D target() const;

	float distanceTarget() const;


	void setRefreshPosition(QVector3D pos);

	bool helicoVisible( ) const;

	void showLineVert(bool );

//	void setPosition ( QVector3D pos);
//	void setTarget ( QVector3D target);

	virtual QUuid dataID() const override;
public	slots:
	void requestTargetChanged(QVector3D  val);
	void setTargetFromRep(QVector3D target);

	void requestPosChanged(QVector3D  val);
	void setPositionFromRep(QVector3D pos);

	void requestPosXChanged(float x);
	void requestPosYChanged(float y);

	void showHelico(bool b);

	void setDistanceTargetFromRep(float );



signals:
	void posChanged(QVector3D);
	void positionToRepChanged(QVector3D);

	void targetChanged(QVector3D);

	void helicoShowed(bool);
	void distanceTargetChanged(float);


private:

	QVector3D m_position;
	QVector3D m_target;

	float m_distanceTarget;


	bool m_helicoVisible= false;

	DatasetRep* m_rep;
	ViewQt3D* m_view3d;
};


#endif

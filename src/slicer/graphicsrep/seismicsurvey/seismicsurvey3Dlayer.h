#ifndef SeismicSurvey3DLayer_H
#define SeismicSurvey3DLayer_H

#include "graphic3Dlayer.h"
#include <QMatrix4x4>

class SeismicSurveyRep;

namespace Qt3DCore {
	class QEntity;
	class QTransform;
}

class SeismicSurvey3DLayer: public Graphic3DLayer {
Q_OBJECT
public:
	SeismicSurvey3DLayer(SeismicSurveyRep *rep,QWindow * parent, Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~SeismicSurvey3DLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void zScale(float val) override;


protected slots:
	virtual void refresh() override;
protected:
	Qt3DCore::QEntity *m_volumeEntity;
	SeismicSurveyRep *m_rep;
	Qt3DCore::QTransform *m_transform;
	QMatrix4x4 m_transformMatrixOri;

};

#endif

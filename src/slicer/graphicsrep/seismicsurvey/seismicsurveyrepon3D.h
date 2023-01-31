#ifndef SeismicSurveyRepOn3D_H
#define SeismicSurveyRepOn3D_H

#include <QObject>
#include <QMap>
#include "seismicsurveyrep.h"
#include "idatacontrolerholder.h"
class SeismicSurvey3DLayer;

class SeismicSurveyRepOn3D: public SeismicSurveyRep {
Q_OBJECT
public:
	SeismicSurveyRepOn3D(SeismicSurvey *survey, AbstractInnerView *parent = 0);
	virtual ~SeismicSurveyRepOn3D();

	virtual bool canBeDisplayed() const override {
		return true;
	}
	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;
	virtual TypeRep getTypeGraphicRep() override;
private:
	SeismicSurvey3DLayer *m_layer;
};

#endif

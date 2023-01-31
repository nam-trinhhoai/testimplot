#include "seismicsurveyrepon3D.h"
#include "seismicsurvey.h"
#include "seismicsurvey3Dlayer.h"

SeismicSurveyRepOn3D::SeismicSurveyRepOn3D(SeismicSurvey *survey, AbstractInnerView *parent) :
	SeismicSurveyRep(survey,parent) {
	m_layer = nullptr;
}

SeismicSurveyRepOn3D::~SeismicSurveyRepOn3D() {
	if (m_layer!=nullptr) {
		delete m_layer;
	}
}

Graphic3DLayer * SeismicSurveyRepOn3D::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera){
	if (m_layer == nullptr)
		m_layer = new SeismicSurvey3DLayer(this,parent,root,camera);
	return m_layer;
}

QWidget* SeismicSurveyRepOn3D::propertyPanel() {
	return nullptr;
}
GraphicLayer * SeismicSurveyRepOn3D::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)
{
	return nullptr;
}

AbstractGraphicRep::TypeRep SeismicSurveyRepOn3D::getTypeGraphicRep(){
	return AbstractGraphicRep::Image3D;
}



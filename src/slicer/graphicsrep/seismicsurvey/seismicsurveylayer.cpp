#include "seismicsurveylayer.h"
#include <QGraphicsScene>
#include "seismicsurveyrep.h"
#include "qglimagegriditem.h"
#include "seismicsurvey.h"
#include "affine2dtransformation.h"

SeismicSurveyLayer::SeismicSurveyLayer(SeismicSurveyRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent) :
GraphicLayer(scene, defaultZDepth) {
	m_rep=rep;

	m_item = new QGLImageGridItem(((SeismicSurvey *)rep->data())->ijToXYTransfo(),parent);
	m_item->setColor(Qt::yellow);
	m_item->setZValue(defaultZDepth);
}

SeismicSurveyLayer::~SeismicSurveyLayer() {
}
void SeismicSurveyLayer::show()
{
	m_scene->addItem(m_item);

}
void SeismicSurveyLayer::hide()
{
	m_scene->removeItem(m_item);
}

QRectF SeismicSurveyLayer::boundingRect() const
{
	return dynamic_cast<SeismicSurvey *>(m_rep->data())->ijToXYTransfo()->worldExtent();
}


void SeismicSurveyLayer::refresh() {
	m_item->update();
}


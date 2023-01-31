#ifndef SeismicSurveyLayer_H
#define SeismicSurveyLayer_H

#include "graphiclayer.h"

class QGraphicsItem;
class QGLImageGridItem;
class SeismicSurveyRep;

class SeismicSurveyLayer : public GraphicLayer{
	  Q_OBJECT
public:
    SeismicSurveyLayer(SeismicSurveyRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent);
	virtual ~SeismicSurveyLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRectF boundingRect() const override;

protected slots:
	virtual void refresh() override;

protected:
	QGLImageGridItem *m_item;
	SeismicSurveyRep *m_rep;
};

#endif

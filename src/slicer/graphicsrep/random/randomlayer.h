#ifndef RandomLayer_H
#define RandomLayer_H

#include "graphiclayer.h"
class QGraphicsItem;
class QGLFullCUDAImageItem;
class RandomRep;
class QGLColorBar;

class RandomLayer : public GraphicLayer{
	  Q_OBJECT
public:
	RandomLayer(RandomRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~RandomLayer();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

    void showColorScale(bool val);

    //warning: disconection not done because image suppose deleted
    void updateImage();
public slots:
	virtual void refresh() override;
signals:
    void hidden();
private:
	void updateColorScale();
protected:
	QGLFullCUDAImageItem *m_item;
	RandomRep *m_rep;

	QGLColorBar * m_colorScaleItem;
};

#endif

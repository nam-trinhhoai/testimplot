#ifndef SliceLayer_H
#define SliceLayer_H

#include "graphiclayer.h"
class QGraphicsItem;
class QGLFullCUDAImageItem;
class SliceRep;
class QGLColorBar;

class SliceLayer : public GraphicLayer{
	  Q_OBJECT
public:
    SliceLayer(SliceRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~SliceLayer();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

    void showColorScale(bool val);
public slots:
	virtual void refresh() override;
signals:
   void hidden();
private:
	void updateColorScale();
protected:
	QGLFullCUDAImageItem *m_item;
	SliceRep *m_rep;

	QGLColorBar * m_colorScaleItem;
};

#endif

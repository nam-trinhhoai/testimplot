#ifndef ComputationOperatorDatasetLayer_H
#define ComputationOperatorDatasetLayer_H

#include "graphiclayer.h"
class QGraphicsItem;
class QGLFullCUDAImageItem;
class ComputationOperatorDatasetRep;
class QGLColorBar;

class ComputationOperatorDatasetLayer : public GraphicLayer{
	  Q_OBJECT
public:
	  ComputationOperatorDatasetLayer(ComputationOperatorDatasetRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~ComputationOperatorDatasetLayer();

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
	ComputationOperatorDatasetRep *m_rep;

	QGLColorBar * m_colorScaleItem;
};

#endif

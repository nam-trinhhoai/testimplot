#ifndef VideoLayerLayer_H
#define VideoLayerLayer_H

#include "graphiclayer.h"

class QGraphicsVideoItem;
class QGraphicsItem;
class VideoLayerRep;

class VideoLayerLayer : public GraphicLayer{
	  Q_OBJECT
public:
	VideoLayerLayer(VideoLayerRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~VideoLayerLayer();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

public slots:
	virtual void refresh() override;

protected:
	QGraphicsVideoItem *m_item;
	VideoLayerRep *m_rep;
};

#endif

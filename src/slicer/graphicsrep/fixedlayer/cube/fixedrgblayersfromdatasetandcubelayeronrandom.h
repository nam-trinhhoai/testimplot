#ifndef FixedRGBLayersFromDatasetAndCubeLayerOnRandom_H
#define FixedRGBLayersFromDatasetAndCubeLayerOnRandom_H

#include "graphiclayer.h"
#include "sliceutils.h"

#include <QPointer>
#include <QTransform>

class QAction;
class QGraphicsItem;
class QMenu;
class FixedRGBLayersFromDatasetAndCubeRepOnRandom;
class GraphEditor_MultiPolyLineShape;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;

class FixedRGBLayersFromDatasetAndCubeLayerOnRandom : public GraphicLayer{
	  Q_OBJECT
public:
	FixedRGBLayersFromDatasetAndCubeLayerOnRandom(FixedRGBLayersFromDatasetAndCubeRepOnRandom *rep,
			QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedRGBLayersFromDatasetAndCubeLayerOnRandom();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;
public slots:
	virtual void refresh() override;

protected:
	void actionMenuCreate();

	QPointer<GraphEditor_MultiPolyLineShape> m_polylineShape;
	FixedRGBLayersFromDatasetAndCubeRepOnRandom *m_rep;
	QTransform m_mainTransform;

	QMenu *m_itemMenu = nullptr;
	QAction *m_actionColor = nullptr;
	QAction *m_actionProperties = nullptr;
	QAction *m_actionLocation = nullptr;
};

#endif

#ifndef FixedLayersFromDatasetAndCubeLayerOnRandom_H
#define FixedLayersFromDatasetAndCubeLayerOnRandom_H

#include "graphiclayer.h"
#include "sliceutils.h"

#include <QPointer>
#include <QTransform>

class QAction;
class QGraphicsItem;
class QMenu;
class FixedLayersFromDatasetAndCubeRepOnRandom;
class GraphEditor_MultiPolyLineShape;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;

class FixedLayersFromDatasetAndCubeLayerOnRandom : public GraphicLayer{
	  Q_OBJECT
public:
	FixedLayersFromDatasetAndCubeLayerOnRandom(FixedLayersFromDatasetAndCubeRepOnRandom *rep,
			QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedLayersFromDatasetAndCubeLayerOnRandom();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;
public slots:
	virtual void refresh() override;

protected:
	void actionMenuCreate();

	QPointer<GraphEditor_MultiPolyLineShape> m_polylineShape;
	FixedLayersFromDatasetAndCubeRepOnRandom *m_rep;
	QTransform m_mainTransform;

	QMenu *m_itemMenu = nullptr;
	QAction *m_actionColor = nullptr;
	QAction *m_actionProperties = nullptr;
	QAction *m_actionLocation = nullptr;
};

#endif

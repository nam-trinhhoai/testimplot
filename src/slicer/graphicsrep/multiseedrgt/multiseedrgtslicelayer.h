#ifndef MultiSeedRgtSliceLayer_H
#define MultiSeedRgtSliceLayer_H

#include "graphiclayer.h"
#include "curve.h"

#include <memory>

class QGraphicsScene;
class QGraphicsView;
class QGraphicsItem;
class QGraphicsEllipseItem;

class MultiSeedRgtSliceRep;

class MultiSeedRgtSliceLayer: public GraphicLayer
{
	Q_OBJECT
public:
	MultiSeedRgtSliceLayer(MultiSeedRgtSliceRep* rep, QGraphicsScene *scene, int defaultZDepth, QGraphicsItem* parent = nullptr);
	virtual ~MultiSeedRgtSliceLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRectF boundingRect() const override;
	void refreshPolygons();
	virtual void refresh() override;

protected:
	std::unique_ptr<Curve> m_curveMain;
	std::unique_ptr<Curve> m_curveTop;
	std::unique_ptr<Curve> m_curveBottom;

	std::vector<std::unique_ptr<Curve>> m_curveReference;

	MultiSeedRgtSliceRep* m_rep = nullptr;

	std::vector<QGraphicsEllipseItem*> m_seedsRepresentation;
	QGraphicsView* m_view = nullptr;
};

#endif


#ifndef MultiSeedRandomLayer_H
#define MultiSeedRandomLayer_H

#include "graphiclayer.h"
#include "curve.h"
#include "referencecurvewrapper.h"

#include <memory>

class QGraphicsScene;
class QGraphicsItem;
class QGraphicsEllipseItem;

class MultiSeedRandomRep;

class MultiSeedRandomLayer: public GraphicLayer
{
	Q_OBJECT
public:
	MultiSeedRandomLayer(MultiSeedRandomRep* rep, QGraphicsScene *scene, int defaultZDepth, QGraphicsItem* parent = nullptr);
	virtual ~MultiSeedRandomLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRectF boundingRect() const override;
	void refreshPolygons();
	virtual void refresh() override;

protected:
	void referencesChanged();

	std::unique_ptr<Curve> m_curveMain;
	std::unique_ptr<Curve> m_curveTop;
	std::unique_ptr<Curve> m_curveBottom;

	std::vector<std::shared_ptr<ReferenceCurveWrapper>> m_curveReference;

	MultiSeedRandomRep* m_rep = nullptr;
	QGraphicsItem* m_parent;

	std::vector<QGraphicsEllipseItem*> m_seedsRepresentation;
};

#endif


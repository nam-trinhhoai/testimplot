#ifndef MultiSeedSliceLayer_H
#define MultiSeedSliceLayer_H

#include "graphiclayer.h"
#include "curve.h"
#include "referencecurvewrapper.h"

#include <memory>

class QGraphicsScene;
class QGraphicsView;
class QGraphicsItem;
class QGraphicsEllipseItem;

class FixedLayerFromDataset;
class MultiSeedSliceRep;


class MultiSeedSliceLayer: public GraphicLayer
{
	Q_OBJECT
public:
	MultiSeedSliceLayer(MultiSeedSliceRep* rep, QGraphicsScene *scene, int defaultZDepth, QGraphicsItem* parent = nullptr);
	virtual ~MultiSeedSliceLayer();

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

	MultiSeedSliceRep* m_rep = nullptr;

	std::vector<QGraphicsEllipseItem*> m_seedsRepresentation;
	QGraphicsView* m_view = nullptr;
	QGraphicsItem* m_parent;
};

#endif


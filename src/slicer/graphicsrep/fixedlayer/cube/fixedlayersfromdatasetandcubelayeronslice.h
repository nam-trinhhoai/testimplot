#ifndef FixedLayersFromDatasetAndCubeLayerOnSlice_H
#define FixedLayersFromDatasetAndCubeLayerOnSlice_H

#include "graphiclayer.h"
#include "sliceutils.h"

#include <QPointer>
#include <QTransform>

class QAction;
class QGraphicsItem;
class QMenu;
class FixedLayersFromDatasetAndCubeRepOnSlice;
class QGraphicsScene;
class QGLIsolineItem;
class GraphEditor_MultiPolyLineShape;
class IGeorefImage;
class QTransform;

class FixedLayersFromDatasetAndCubeLayerOnSlice : public GraphicLayer{
	  Q_OBJECT
public:
	FixedLayersFromDatasetAndCubeLayerOnSlice(FixedLayersFromDatasetAndCubeRepOnSlice *rep,SliceDirection dir,
			int startValue,QGraphicsScene *scene,
			int defaultZDepth,QGraphicsItem *parent);
	virtual ~FixedLayersFromDatasetAndCubeLayerOnSlice();

	void setSliceIJPosition(int imageVal);

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;
    bool isShapeSelected();


public slots:
	virtual void refresh() override;

protected:
	void actionMenuCreate();
	void mouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) override;

	FixedLayersFromDatasetAndCubeRepOnSlice *m_rep;
	QTransform m_mainTransform;

	QPointer<GraphEditor_MultiPolyLineShape> m_polylineShape;
	QMenu *m_itemMenu = nullptr;
	QAction *m_actionColor = nullptr;
	QAction *m_actionProperties = nullptr;
	QAction *m_actionLocation = nullptr;
};

#endif

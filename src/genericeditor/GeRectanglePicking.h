#ifndef GeRectanglePicking_H
#define GeRectanglePicking_H

#include <QObject>
#include <QColor>
#include <QPointF>
//#include "view2d/qextendablegraphicsscene.h"
//#include "view2d/sceneextension.h"
#include "pickingtask.h"
#include "GeObjectId.h"
#include "GeGlobalParameters.h"

class QGraphicsSceneMouseEvent;
class QGraphicsScene;
class QKeyEvent;
class QPolygonF;
class GeRectanglePicking;
class QGraphicsItem;

class SyncViewer2d;
class MtGraphicsPolygon;

class GeRectanglePicking : public PickingTask {
    Q_OBJECT

    Q_PROPERTY(int currentAction READ currentAction WRITE setCurrentAction NOTIFY currentActionChanged)
    Q_PROPERTY(QPointF previousPosition READ previousPosition WRITE setPreviousPosition NOTIFY previousPositionChanged)

public:
    GeRectanglePicking(/*data::MtVaData *vaData,*/
    		GeGlobalParameters& globalParameters, QObject *parent = 0);
    ~GeRectanglePicking();

    enum ActionTypes {
        DefaultType,
        LineType,
		PolygonType,
        RectangleType,
        EllipseType,
        PenType,
        SelectionType
    };

    int currentAction() const;
    QPointF previousPosition() const;

    void setCurrentAction(const int type);
    void setCurrentAction2(const int type, int objectIndex, QColor& color);
    void setPreviousPosition(const QPointF previousPosition);

	virtual void initCanvas(QGraphicsScene* canvas);
	virtual void releaseCanvas(QGraphicsScene* canvas);

	virtual void mouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;
	virtual void mousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;

	void cleanScene();
	void cleanObjectGraphicsOnSlice(int objectId);
	void polygonPointAdded(int objectId, int indexAdded, double factor);
	void polygonPointDeleted(int objectId, int index);
	void setSlice(int slice);
	void setColor(const QColor& c);

	void objectSelectedFromItem(int ojectId);
	void propagSelectionOnGraphics(int objectId);

	void drawAllGraphicsItems();
	void refreshObjectDisplay(int objectId);
	void rectDisplay(int objectId, int currentSlice, bool removeBefore);

	void refreshOrCreateItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere,
			bool removeBefore);
	void refreshOrCreateItem(int objectId, QPolygonF& newRect, QColor& color, bool pickedHere,
			bool removeBefore);
	void refreshOrCreateEllipseItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere,
			bool removeBefore);

    void createItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere);
    void createItem(int objectId, QPolygonF& newRect, QColor& color, bool pickedHere);
    void createEllipseItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere);

signals:
    void previousPositionChanged();
    void currentActionChanged(int);
    void signalSelectItem(QGraphicsItem *item);
    void signalNewSelectItem(QGraphicsItem *item, int indexColor);
    void signalRemoveItem(QGraphicsItem *item);
    void rectangleChanged(const QRectF& newRect, int objectIndex, GeRectanglePicking* originGraphicsPointer);
    void polygonChanged(const QPolygonF& newPolygon, int objectIndex, GeRectanglePicking* originGraphicsPointer);
    void ellipseChanged(const QRectF& newRect, int objectIndex, GeRectanglePicking* originGraphicsPointer);

    void objectSelectedSignal(int objectIndex);

protected:

    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);

public slots:
    void slotMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
    void deselectItems();
    void checkSelection();

	SyncViewer2d* getViewer() const {
		return viewer;
	}

private:
	QGraphicsScene* m_canvas = nullptr;
    QGraphicsItem *m_currentItem;
    int m_currentAction;
    QColor m_currentColor;
    GeObjectId currentObjectId;
    int m_previousAction;
    QPointF m_previousPosition;
    bool m_leftMouseButtonPressed;
    int m_indexColor = -1;
    static const GeObjectId selectionNilObjectId;
    SyncViewer2d *viewer;
    GeGlobalParameters& globalParameters;
    //data::MtVaData *vaData;
    bool polygonVolatyOn = false;
};

#endif // GeRectanglePicking_H

#ifndef GraphicsPointerExt_H
#define GraphicsPointerExt_H

#include <QObject>
#include <QPolygonF>
#include <QColor>
#include <QList>
#include <QEvent>
#include <QGraphicsItem>
//#include "view2d/qextendablegraphicsscene.h"
//#include "view2d/sceneextension.h"
#include "GeObjectId.h"
#include "GeGlobalParameters.h"
#include "pickingtask.h"

class Abstract2DInnerView;

class QGraphicsScene;
class QGraphicsItem;
class QGraphicsSceneMouseEvent;
class QKeyEvent;

//class MtGraphicsPolygon;

enum DrawingType {
	Rectangle,
	Polygon,
	PolyLine,
	Ellipse,
	NoType
};

class SceneEventBlocker : public QObject {
public:
	SceneEventBlocker(QList<QEvent::Type> blockList, QObject* parent=0);
	virtual ~SceneEventBlocker();

//	virtual QRectF boundingRect() const override;
//	virtual void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*) override;

protected:
	virtual bool eventFilter(QObject* item, QEvent* event) override;
private:
	QList<QEvent::Type> m_blockTypes;
};

class GraphicsPointerExt : public PickingTask
{
    Q_OBJECT

    Q_PROPERTY(int currentAction READ currentAction WRITE setCurrentAction NOTIFY currentActionChanged)
    Q_PROPERTY(QPointF previousPosition READ previousPosition WRITE setPreviousPosition NOTIFY previousPositionChanged)

public:
    explicit GraphicsPointerExt(/*data::MtVaData *vaData,*/ GeGlobalParameters& globalParameters, Abstract2DInnerView *parent = 0);
    ~GraphicsPointerExt();

    enum ActionTypes {
        DefaultType,
        LineType,
		PolygonType,
        RectangleType,
        EllipseType,
		PolyLineType,
        PenType,
        SelectionType
    };

    int currentAction() const;
    QPointF previousPosition() const;

    void setCurrentAction(const int type);
    void setCurrentAction2(const int type, int objectIndex, QColor& color);
    void setPreviousPosition(const QPointF previousPosition);

//	virtual void initCanvas(QGraphicsScene* canvas) override;
//	virtual void releaseCanvas(QGraphicsScene* canvas) override;

	void cleanScene();
	void cleanObjectGraphicsOnSlice(int objectId);
	void polygonPointAdded(int objectId, int indexAdded, double factor);
	void polygonPointDeleted(int objectId, int index);
	void polyLinePointAdded(int objectId, int indexAdded, double factor);
	void polyLinePointDeleted(int objectId, int index);
	void setSlice(int slice);
	void setColor(const QColor& c);

	void objectSelectedFromItem(int ojectId);
	void propagSelectionOnGraphics(int objectId);

	void drawAllGraphicsItems();
	void refreshObjectDisplay(int objectId);
	void rectDisplay(int objectId, int currentSlice, bool removeBefore);
	void polygonDisplay(int objectId, int currentSlice, bool removeBefore);
	void polyLineDisplay(int objectId, int currentSlice, bool removeBefore);
	void ellipseDisplay(int objectId, int currentSlice, bool removeBefore);

	void refreshOrCreateItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere,
			bool removeBefore);
	void refreshOrCreateItem(int objectId, QPolygonF& newRect, QColor& color, bool pickedHere,
			bool removeBefore);
	void refreshOrCreateEllipseItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere,
			bool removeBefore);

    void createItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere);
    void createItem(int objectId, QPolygonF& newRect, QColor& color, bool pickedHere);
    void createPolyLineItem(int objectId, QPolygonF& newRect, QColor& color, bool pickedHere);
    void createEllipseItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere);

    void setDefaultZ(int newZ);
    int defaultZ() const;

signals:
    void previousPositionChanged();
    void currentActionChanged(int);
    void endEditionItem(QGraphicsItem *item);
    void signalSelectItem(QGraphicsItem *item);
    void signalNewSelectItem(QGraphicsItem *item, int indexColor);
    void signalRemoveItem(QGraphicsItem *item);
    void rectangleChanged(const QRectF& newRect, /*int objectIndex,*/ GraphicsPointerExt* originGraphicsPointer);
    void polygonChanged(const QPolygonF& newPolygon, int objectIndex, GraphicsPointerExt* originGraphicsPointer);
    void polyLineChanged(const QPolygonF& newPolygon, int objectIndex, GraphicsPointerExt* originGraphicsPointer);
    void ellipseChanged(const QRectF& newRect, int objectIndex, GraphicsPointerExt* originGraphicsPointer);

    void polygonPointAddedSignal(int modifiedObjectId, int index, double factor, /*int slice,*/ GraphicsPointerExt* originGraphicsPointer);
    void polygonPointDeletedSignal(int modifiedObjectId);
    void polyLinePointAddedSignal(int modifiedObjectId, int index, double factor, /*int slice,*/ GraphicsPointerExt* originGraphicsPointer);
    void polyLinePointDeletedSignal(int modifiedObjectId);
    void objectSelectedSignal(int objectIndex);

protected:
	virtual void mouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;
	virtual void mousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;
	virtual void mouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;
	virtual void mouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;

//    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
//    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
//    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
//    void keyPressEvent(QKeyEvent *event) override;

public slots:
    void slotMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
    void deselectItems();
    void checkSelection();

	Abstract2DInnerView* getViewer() const {
		return viewer;
	}

private:
//    QExtendableGraphicsScene* canvas = nullptr;
	QGraphicsScene* canvas = nullptr;
    QGraphicsItem *currentItem;
    int m_currentAction;
    QColor m_currentColor;
//    GeObjectId currentObjectId;
    int m_previousAction;
    QPointF m_previousPosition;
    bool m_leftMouseButtonPressed;
    int m_indexColor = -1;
//    static const GeObjectId selectionNilObjectId;
//    SyncViewer2d *viewer;
    Abstract2DInnerView* viewer;
    GeGlobalParameters globalParameters;
//    data::MtVaData *vaData;
    QPolygonF polygon;
    bool polygonOn = false;
    bool polygonVolatyOn = false;
    QPolygonF qPolygon;
    DrawingType m_drawingType;

    int m_defaultZ = 0;

    SceneEventBlocker m_eventBlocker;
};


//} /* namespace view2d */
//} /* namespace gui */
//} /* namespace murat */
#endif // GraphicsPointerExt_H

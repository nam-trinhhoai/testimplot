#ifndef __GRAPHEDITOR_MULTIPOLYLINESHAPE__
#define __GRAPHEDITOR_MULTIPOLYLINESHAPE__

class QGraphicsSimpleTextItem;
class LineLengthText;
class GraphicSceneEditor;
#include "GraphEditor_Path.h"

class GraphEditor_MultiPolyLineShape : public GraphEditor_Path
{
	Q_OBJECT
public:
	enum { NOVALUE = -9999 };
	// GraphEditor_MultiPolyLineShape(QPolygonF polygon, QPen pen, QBrush brush, QMenu*, GraphicSceneEditor* scene=nullptr, bool issClosedCurved=false);
	GraphEditor_MultiPolyLineShape();
    ~GraphEditor_MultiPolyLineShape();

    virtual void setPolygon(const QPolygonF &poly);
    // virtual void setPolygon(const std::vector<QPolygonF> &poly);
    GraphEditor_MultiPolyLineShape* clone();
	QVector<QPointF> SceneCordinatesPoints() override;
	QVector<QPointF> ImageCordinatesPoints() override;
	void setDrawFinished(bool value) override;
	bool checkClosedPath();

	void insertNewPoints(QPointF) override;

	QVector<QPointF> getKeyPoints() override;
	signals:
	void currentIndexChanged(int);
	// void polygonChanged(QVector<QPointF>,bool);

	public slots:
	void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void GrabberMouseReleased(QGraphicsItem *signalOwner);
	void moveGrabber(int, int dx, int dy);
	void positionGrabber(int, QPointF);
	void polygonChanged1();
	void polygonResize(int,int);
	void setDisplayPerimetre(bool value) { m_displayPerimetre = value; }

	public:
	void wheelEvent(QGraphicsSceneWheelEvent *event) override;
	void setRotation(qreal angle) { QGraphicsItem::setRotation(0.0); }
	bool sceneEvent(QEvent *event);
	void setNoValue(float val) { m_noValue = val; }
	bool setSelect(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
	bool isSelected();

	protected slots:
		bool eventFilter(QObject* watched, QEvent* ev) override;


private:
	int m_searchArea = 1;
	bool m_displayPerimetre = true;
	std::vector<QPainterPath> m_painterPath;
	float m_noValue = 0.0f;
	std::vector<std::vector<int>> qPolygonXSplit(QPolygon& polyIn);
	std::vector<QPolygonF> qPolygonSplit(QPolygonF& polyIn0);
	void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
	void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
	void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
	int getTabIndex(float x);
	void inputMethodEvent(QInputMethodEvent *event) override;
	bool isSelectable(double worldX,double worldY);
	QPainterPath shape() const override;
	QPolygonF m_polygonShape;
	QPolygonF polygonShapeCreate(QPolygonF poly);




protected:
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    virtual void setPolygonFromMove(const QPolygonF &poly);
    virtual void showGrabbers();
	void calculatePerimeter();
};


#endif

/*
 * GraphEditor_LineShape.h
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GraphEditor_LineShape_H_
#define SRC_GENERICEDITOR_GraphEditor_LineShape_H_

#include <QObject>
#include <QPainter>
#include <QGraphicsLineItem>
#include <QStyleOptionGraphicsItem>

#include "GraphEditor_Item.h"
#include "GraphEditor_ItemInfo.h"

class QGraphicsItem;
class GraphEditor_GrabberItem;
class QGraphicsSceneMouseEvent;

enum eLineFlags{
	E_Flag_None =0x00,
	E_Flag_Ortho = 0x01
};

class GraphEditor_LineShape : public QObject, public QGraphicsLineItem, public GraphEditor_Item, public GraphEditor_ItemInfo
{
	Q_OBJECT

public:
	GraphEditor_LineShape(QLineF line, QPen pen, QMenu*,GraphicSceneEditor* scene=nullptr);
	~GraphEditor_LineShape();

	GraphEditor_LineShape* clone();
    virtual void setLine(const QLineF &line);

    virtual void setRandomView(RandomLineView *pRandView) override;

    void ContextualMenu(QPoint)override;
	QVector<QPointF> SceneCordinatesPoints() override;
	QVector<QPointF> ImageCordinatesPoints() override;
	void setOrthogonal(GraphEditor_ItemInfo*);
	GraphEditor_ItemInfo *getOrthogonal();
public slots:

	void FirstPointMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void LastPointMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void updateOrthoWidthline(double iLength);
	void updateOrtholine(QPointF point);
	void refreshOrtholine(QVector3D, QPointF);

	private slots:
	void resetRandomLineView();
signals:
    void orthogonalUpdated(QPolygonF);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    virtual void setLineFromMove(const QLineF &line);
    void updateGrabbersPosition();
    void createGrabbers();
    void setGrabbersVisibility(bool visible);
    void calculateLineWidth();
    GraphEditor_ItemInfo* m_OrthoItem;
    GraphEditor_GrabberItem *grabberList[2];


    bool first = true;
   // QPointF m_pts;
};




#endif /* SRC_GENERICEDITOR_GraphEditor_LineShape_H_ */

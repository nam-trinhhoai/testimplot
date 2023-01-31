/*
 * GraphEditor_ListBezierPath.h
 *
 *  Created on: Mai 18, 2022
 *      Author: l1049100
 */

#ifndef SRC_GENERICEDITOR_GRAPHEDITOR_LISTBEZIERPATH_H_
#define SRC_GENERICEDITOR_GRAPHEDITOR_LISTBEZIERPATH_H_

class QGraphicsSimpleTextItem;
class LineLengthText;
class GraphicSceneEditor;
#include "GraphEditor_Path.h"
#include "PointCtrl.h"


class GraphEditor_ListBezierPath : public GraphEditor_Path
{
	Q_OBJECT
public:
	GraphEditor_ListBezierPath(QPolygonF polygon, QPen pen, QBrush brush, QMenu*,
			GraphicSceneEditor* scene=nullptr, bool issClosedCurved=false, QColor color =Qt::yellow);

	GraphEditor_ListBezierPath(QVector<PointCtrl> listeCtrl, QPen pen, QBrush brush, QMenu* itemMenu,
			GraphicSceneEditor* scene, bool isClosedCurved,QColor color =Qt::yellow);
    ~GraphEditor_ListBezierPath();


    void setPathCurrent(QPainterPath painter);
    virtual void setPolygon(const QPolygonF &poly);
    void setPolygonTangent(const QPolygonF &poly);

    void setPolygone(const QPolygonF &polygon);
    GraphEditor_ListBezierPath* clone();
	QVector<QPointF> SceneCordinatesPoints() override;
	QVector<QPointF> ImageCordinatesPoints() override;
	void setDrawFinished(bool value) override;
	bool checkClosedPath();

	void insertNewPoints(QPointF) override;

	QPointF getPosition(float t);
	QPointF getNormal(float t);

	QVector<PointCtrl> GetListeCtrls(){
			 return m_listePtCtrls;
		}


	void showGrabberCurrent();
	QVector<QPointF> getPointInterpolated();

	QVector<QPointF> getKeyPoints() override;


	void createCornerGrabbers();
	void setPositionCornerGrabbers();
	void setCornerGrabbersVisibility(bool);



	void reinitSelect(bool b)
	{
		qDebug()<<"-->reinitSelect  "<<b;
		m_AlreadySelected = b;
	}

	float getPrecision();
	void setPrecision(float value);

	void setColor(QColor );

	QColor getColor(){ return m_penYellow.color();}

	void restoreState(QPolygonF poly, bool isClosedCurve);






signals:
	void currentIndexChanged(int);
	void polygonChanged(QVector<PointCtrl>,QVector<QPointF>,bool);
	void polygonChanged2(GraphEditor_ListBezierPath*);



	public slots:
	void cornerGrabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void grabberCtrlMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void GrabberMouseReleased(QGraphicsItem *signalOwner);
	void moveGrabber(int, int dx, int dy);
	void positionGrabber(int, QPointF);
	void polygonChanged1();
	void polygonResize(int,int);

	void selectGrabber(QGraphicsItem *signalOwner);
	void GrabberDeleted(QGraphicsItem *signalOwner);
protected:

    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    virtual void setPolygonFromMove(const QPolygonF &poly);
    virtual void showGrabbers();
	void calculatePerimeter();

	void showGrabbersCtrl(bool recomputeTangente=true);

	void addGrabbersCtrl(int i, PointCtrl pts);

	void hideGrabberCurrent();

	QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;



private:
	void refreshCurve();
	void resizeLeftBis(qreal dx);
	void resizeRightBis(qreal dx);
	void resizeBottomBis(qreal dy);
	void resizeTopBis(qreal heightOffset);

	QPainterPath shape() const override;
	double m_scale = .1;

	//QVector<QPointF> m_listeNoeuds;
	//QVector<QPointF> m_listeControls;
	float  m_precision =19.0f;

	QPainterPath pathCurrent;

	QVector<PointCtrl> m_listePtCtrls;

	QList<GraphEditor_GrabberItem *> m_grabbersCtrl;

	int m_indexCurrentCrtl = -1;
	int m_lastIndexCurrent = -2;

	QPen m_penGreen;
	QPen m_penRed;
	QPen m_penYellow;

	float m_coefTangent = 0.2f;

	GraphEditor_GrabberItem *m_cornerGrabber[8];

	bool m_selectStart=true;

	bool showtang = false;
};


#endif /* SRC_GENERICEDITOR_GRAPHEDITOR_LISTBEZIERPATH_H_ */

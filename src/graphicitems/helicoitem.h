#ifndef HelicoItem_H
#define HelicoItem_H

#include <QObject>
#include <QVector>
#include <QVector3D>
#include <QGraphicsSvgItem>
#include <QGraphicsItem>
#include <QGraphicsSceneHoverEvent>
#include <QGraphicsLineItem>
#include <QGraphicsEllipseItem>
#include <QGraphicsSceneEvent>
#include <QPen>
#include "cameraparameterscontroller.h"
class CameraParametersController;

class HelicoItem : public QGraphicsObject
{
	Q_OBJECT
public:
	HelicoItem(CameraParametersController* ctrl, QGraphicsItem *parent);
	~HelicoItem();

	//bool eventFilter(QObject* watched, QEvent* ev) override;
	QRectF boundingRect()const override;
	void paint(QPainter* painter,const QStyleOptionGraphicsItem* option, QWidget *widget) override;

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant & value) override;


	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;

	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;

	//  void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
	//  void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);

	  void wheelEvent(QGraphicsSceneWheelEvent *event);
public slots:
	void positionChanged();
	void transformItemZoomScale(QVector3D position,QGraphicsScene* scene= nullptr);
	void refreshItemZoomScale();
	void showHelico(bool);
	void targetChanged();
	void positionFromCtrl(QVector3D );
	void distanceTargetChanged(float);
	void rotateWheel(float coef);


void test( QGraphicsSceneHoverEvent *event);



private:
	QVector3D moyenneTab();
	double computeScale(QGraphicsScene* scene);

	QGraphicsLineItem * m_lineItem;
	QGraphicsSvgItem* m_helicoItem;
	QGraphicsEllipseItem* m_targetItem;

	CameraParametersController *m_controler;

	float m_posX, m_posY;
	bool m_lock;
	QPointF m_lastPosition;

	QVector<QVector3D> m_tabDirection;
	int m_indexTab;
	int m_sizeTab = 25;

	float m_angleWheel=0.0f;
	bool m_mouseOnHelico = false;
	bool m_helicoVisible;
};

#endif

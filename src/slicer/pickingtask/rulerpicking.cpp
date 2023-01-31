/*
 * RulerPicking.cpp
 *
 *  Created on: mars 2019
 *      Author: l0222891
 */

#include "rulerpicking.h"

#include <QPainter>
#include <QGraphicsSceneMouseEvent>
#include <QKeyEvent>
#include <QDebug>
#include <QFontMetrics>
#include <QtMath>
#include <QGraphicsScene>

bool debugGS =  true;

class RulerSimpleTextItem: public QGraphicsSimpleTextItem {
public:
	RulerSimpleTextItem(QGraphicsItem* parent) :
			QGraphicsSimpleTextItem(parent) {
		setFlags(ItemIgnoresParentOpacity | ItemIgnoresTransformations);
	}
	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
		//painter->translate(15, 5);
		qreal ang1 = this->rotation();
		QRectF bRect = boundingRect();
		double dcos = qCos(qDegreesToRadians(ang1));
		double dsin = qSin(qDegreesToRadians(ang1));
		qDebug() << "ang= " << ang1 << " Bounding width=" << bRect.width() << " Bounding Height= " << bRect.height()
				<< "Pos= " << pos().x()  << "/" << pos().y() << " Cos=" << dcos << " Sin=" << dsin;
		double deltaX = - (bRect.width() * dcos)/2;
		double deltaY = (debugGS) ? 0 : - (bRect.width() * dsin)/2;
		painter->translate(deltaX, deltaY);
		painter->fillRect(boundingRect().adjusted(-3, -3, 3, 3), Qt::black);
		QGraphicsSimpleTextItem::paint(painter, option, widget);
	}
};



RulerPicking::RulerPicking(QObject *parent, int thickness): PickingTask( parent) {
	this->thickness = 3;

	line.setFlag(QGraphicsItem::ItemIsMovable, false); //in fact it is but only by the code
	line.setFlag(QGraphicsItem::ItemIsSelectable, false);
	line.setFlag(QGraphicsItem::ItemSendsGeometryChanges);
	line.setFlag(QGraphicsItem::ItemSendsScenePositionChanges);
	//line.setBrush(Qt::red);
	//line.setPen(Qt::NoPen);
	QPen linePen(Qt::green, thickness);
	linePen.setCosmetic(true);
	line.setPen(linePen);
	line.setOpacity(0.5);
	line.setZValue(10000);

	text = new RulerSimpleTextItem(nullptr);
	text->setPen(QPen(Qt::red, 0));
	text->setFont(QFont("Arial", 12, QFont::Bold));
	text->setOpacity(0.5);
	text->setZValue(10000);
}

RulerPicking::~RulerPicking() {
	delete text;
}

/**
 * Extension methods
 */
void RulerPicking::mouseMoved(
		double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) {
	if (isElasticMove) {
		QPen linePen(Qt::green, thickness);
		linePen.setCosmetic(true);
		line.setPen(linePen);
		linePen.setCosmetic(true);
		current = line.line();
		current.setP2( QPointF(worldX, worldY) );
		line.setLine(current);
		line.setVisible(true);

		QString text1 = QString::number(current.length());

		//QRect boundingText = metrics.boundingRect(text1);
		QRectF boundingText = text->boundingRect();
		text->setBrush(QColor(246, 230, 0));
		text->setText(text1);
		float lineAngle = current.angle();
		QPointF textPos;
		{
			QPointF p1 = current.p1();
			QPointF p2 = current.p2();
			if ( lineAngle > 90 &&lineAngle < 260) {  // Right to left line
				lineAngle -= 180;
				textPos = current.center();
			} else {  // Left to right line
//				double deltaLineX = p2.x() - p1.x();
//				double deltaTextX = boundingText.width();
//				double beginTextX = p1.x() + (deltaLineX - deltaTextX)/2;
//				double deltaLineY = p2.y() - p1.y();
//				double deltaTextY = boundingText.height();
//				double beginTextY = p1.y() + (deltaLineY - deltaTextY)/2;
//				textPos = QPointF(beginTextX, beginTextY);
				//lineAngle -= 180;
				textPos = current.center();
			}
		}
		text->setPos(textPos);
		text->setVisible(true);
		text->setRotation(lineAngle);
		//qDebug() << "mouse moved " << current.x1() << " " << current.y1() << current.x2() << current.y2() <<
		//		 center.x() << center.y() << " LEN: " << text1;

		//emit linePointed(*this, line.line());
	}
}



void RulerPicking::mousePressed(
		double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) {
	if ( button == Qt::LeftButton) {
		if (!isElasticMove) {
			current = line.line();
			current.setP1( QPointF(worldX, worldY) );
			current.setP2( QPointF(worldX, worldY) ); //add twice so the last one is going to be moved
			line.setLine(current);
			//qDebug() << "mousePressEvent P1 "  << current.x1() << " " << current.y1() << current.x2() << current.y2();
			isElasticMove = true;
		} else {
			current = line.line();
			current.setP2(QPointF(worldX, worldY));
			line.setLine( current);
			line.setVisible(true);
			emit linePointed(*this, line.line());
			//qDebug() << "mousePressEvent P2 " << current.x1() << " " << current.y1() << current.x2() << current.y2();
			//setEnabled(false);
			//setEnabled(true);

			isElasticMove = false;
		}
	}
}

//void RulerPicking::keyReleaseEvent(QKeyEvent *keyEvent) {
//	//	Disrectly in meu shortcut
//}
//
void RulerPicking::initCanvas(QGraphicsScene* canvas) {

	canvas->addItem(&line);
	canvas->addItem(text);
}

void RulerPicking::releaseCanvas(QGraphicsScene* canvas) {

	canvas->removeItem(&line);
	canvas->removeItem(text);
}
//
//void RulerPicking::setEnabled(bool enabled) {
//	line.setVisible(enabled);
//	text->setVisible(enabled);
//
//	if (!enabled) {
//		current = QLineF();
//		line.setLine(current); //we clean the polygon
//		QPen linePen(Qt::green, thickness);
//		linePen.setCosmetic(true);
//		line.setPen(linePen);
//		QString currentT = QString();
//		text->setText(currentT);
//	}
//
//	SceneExtension::setEnabled(enabled);
//}

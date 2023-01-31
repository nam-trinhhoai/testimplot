/*
 * LineLengthText.h
 *
 *  Created on: Dec 30, 2021
 *      Author: l1046262
 */

#ifndef SRC_GRAPHICEDITOR_LINELENGTHTEXT_H_
#define SRC_GRAPHICEDITOR_LINELENGTHTEXT_H_

#include <QGraphicsSimpleTextItem>
#include <QPainter>
#include <QtMath>


class LineLengthText: public QGraphicsSimpleTextItem {
public:
	LineLengthText(QGraphicsItem* parent) :
		QGraphicsSimpleTextItem(parent)
	{
		setFlags(ItemIgnoresParentOpacity | ItemIgnoresTransformations);
		setPen(QPen(Qt::yellow));
		setBrush(QBrush(Qt::yellow));
		setFont(QFont("Arial", 9, QFont::Normal));
		setZValue(6000);
		setVisible(true);
	}

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
		painter->setRenderHint(QPainter::Antialiasing,true);
	//	painter->setRenderHint(QPainter::HighQualityAntialiasing, true);
		qreal ang1 = this->rotation();
		QRectF bRect = boundingRect();
		double dcos = qCos(qDegreesToRadians(ang1));
		double dsin = qSin(qDegreesToRadians(ang1));
		double deltaX = - (bRect.width() * dcos)/2;
		double deltaY = 0;
		painter->translate(deltaX, deltaY);
		painter->setBrush(brush());
		QColor clr(Qt::black);
		clr.setAlpha(122);
		QPen borderPen(Qt::white);
		borderPen.setWidth(2);
		borderPen.setCosmetic(true);
		painter->setPen(borderPen);
		painter->drawRect(boundingRect().adjusted(4, 4, 4, 4));
		painter->fillRect(boundingRect().adjusted(4, 4, 4, 4), clr);
		QRectF rect = boundingRect().adjusted(4, 4, 4, 4);
		painter->setFont(font());
		painter->drawText(rect, Qt::AlignCenter, text(),&rect);
		//QGraphicsSimpleTextItem::paint(painter, option, widget);
	}
};


#endif /* SRC_GRAPHICEDITOR_LINELENGTHTEXT_H_ */

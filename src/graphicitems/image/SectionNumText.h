/*
 * SectionNumText.h
 *
 *  Created on: Jan 10, 2022
 *      Author: l1046262
 */

#ifndef SRC_GRAPHICEDITOR_SectionNumText_H_
#define SRC_GRAPHICEDITOR_SectionNumText_H_

#include <QGraphicsSimpleTextItem>
#include <QPainter>
#include <QtMath>


class SectionNumText: public QGraphicsSimpleTextItem {
public:
	SectionNumText(QGraphicsItem* parent) :
		QGraphicsSimpleTextItem(parent)
	{
		setFlags(ItemIgnoresParentOpacity | ItemIgnoresTransformations);
		setPen(QPen(Qt::yellow));
		setBrush(QBrush(Qt::yellow));
		setFont(QFont("Arial", 7, QFont::Normal));
		setZValue(6000);
		setVisible(true);
	}

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
		painter->setRenderHint(QPainter::Antialiasing,true);
	//	painter->setRenderHint(QPainter::HighQualityAntialiasing, true);

		painter->setBrush(brush());
		//QColor clr(Qt::black);
		//clr.setAlpha(122);
		QPen borderPen(pen());
		borderPen.setWidth(2);
		painter->setPen(borderPen);
		painter->setFont(font());
		QRectF rect = boundingRect().translated(-boundingRect().width()-2,-boundingRect().height()/2);
		painter->drawText(rect, Qt::AlignCenter, text(),&rect);

	}
};


#endif /* SRC_GRAPHICEDITOR_SectionNumText_H_ */

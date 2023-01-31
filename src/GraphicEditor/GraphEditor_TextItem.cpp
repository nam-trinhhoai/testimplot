/*
 * GraphEditor_TextItem.cpp
 *
 *  Created on: Oct 27, 2021
 *      Author: l1046262
 */

#include "GraphEditor_TextItem.h"
#include <QDebug>
#include <QTextCursor>
#include <QFont>
#include <QColor>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>

GraphEditor_TextItem::GraphEditor_TextItem(QGraphicsItem *parent)
: QGraphicsTextItem()
{
	setFlag(QGraphicsItem::ItemIsMovable);
	setFlag(QGraphicsItem::ItemIsSelectable);
	setFlag(QGraphicsItem::ItemSendsGeometryChanges);
	setTextInteractionFlags(Qt::TextEditorInteraction);
	setFlag(ItemIgnoresTransformations);
	positionLastTime = QPointF(0, 0);
}

GraphEditor_TextItem* GraphEditor_TextItem::clone() {
	GraphEditor_TextItem* cloned = new GraphEditor_TextItem(nullptr);
	cloned->setPlainText(toPlainText());
	cloned->setFont(font());
	cloned->setTextWidth(textWidth());
	cloned->setDefaultTextColor(defaultTextColor());
	cloned->setPos(scenePos());
	cloned->setZValue(zValue());
	return cloned;
}

QVariant GraphEditor_TextItem::itemChange(GraphicsItemChange change,
		const QVariant &value)
{
	switch(change)
	{
	case QGraphicsItem::ItemSelectedChange:
	{
		if(!value.toBool()) {
			m_AlreadySelected=false;
		}
		break;
	}
	default : break;
	}
	return QGraphicsItem::itemChange(change, value);
}

void GraphEditor_TextItem::focusInEvent(QFocusEvent* event) {
	if (scene()->selectedItems().empty())
	{
		setSelected(true);
	}
	//setTextInteractionFlags(Qt::TextEditorInteraction);
	if (positionLastTime == QPointF(0, 0))
		// initialize positionLastTime to insertion position
		positionLastTime = scenePos();
	QGraphicsTextItem::focusInEvent(event);
}

void GraphEditor_TextItem::focusOutEvent(QFocusEvent *event) {
	if (!m_AlreadySelected)
		setSelected(false);
	//setTextInteractionFlags(Qt::NoTextInteraction);

	if (contentLastTime == toPlainText())
	{
		contentHasChanged = false;
	} else
	{
		contentLastTime = toPlainText();
		contentHasChanged = true;
	}
	QGraphicsTextItem::focusOutEvent(event);
}

void GraphEditor_TextItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {

	m_AlreadySelected = true;

	QGraphicsTextItem::mouseDoubleClickEvent(event);
}

void GraphEditor_TextItem::mousePressEvent(QGraphicsSceneMouseEvent* event) {
	//setTextInteractionFlags(Qt::TextEditorInteraction);

	if (m_AlreadySelected && (event->button() != Qt::LeftButton))
		return;
	m_AlreadySelected = true;

	QGraphicsTextItem::mousePressEvent(event);
}

void GraphEditor_TextItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
	QColor clr(Qt::black);
	clr.setAlpha(122);
	QPen borderPen(Qt::white);
	if ((option->state & QStyle::State_Selected) || m_AlreadySelected)
	{
		borderPen.setWidth(5);
	}
	else
	{
		borderPen.setWidth(2);
	}

	painter->setPen(borderPen);
	painter->setBrush(QBrush(Qt::NoBrush));
	painter->drawRect(boundingRect());
	painter->fillRect(boundingRect().adjusted(3, 3, -3, -3), clr);

	//painter->setFont(font());
	//painter->drawText(boundingRect(),0,"hello");
	QStyleOptionGraphicsItem NewOption(*option);
	NewOption.state &= !QStyle::State_Selected;
	QGraphicsTextItem::paint(painter, &NewOption, widget);
}

void GraphEditor_TextItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
	if (scenePos() != positionLastTime)
	{
		isMoved = true;
	}
	positionLastTime = scenePos();
	QGraphicsTextItem::mouseReleaseEvent(event);
}

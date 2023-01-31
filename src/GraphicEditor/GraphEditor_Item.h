/*
 * GraphEditor_Item.h
 *
 *  Created on: Nov 1, 2021
 *      Author: l1046262
 */

#ifndef SRC_GRAPHICEDITOR_GraphEditor_Item_H_
#define SRC_GRAPHICEDITOR_GraphEditor_Item_H_

#include <QPen>
#include <QMenu>
#include <QGraphicsItem>
#include <QPoint>
#include <QString>
#include <QDebug>

class QGraphicsSimpleTextItem;

class GraphEditor_Item
{
public:
	GraphEditor_Item() {};

	void setID(QString id)
	{
		m_LayerID =id;
	}

	QString getID()
	{
		return m_LayerID;
	}

	bool isResized()
	{
		return m_ItemGeometryChanged;
	}

	void setResized(bool val)
	{
		m_ItemGeometryChanged=val;
	}

	bool isMoved()
	{
		return m_ItemMoved;
	}

	bool setMoved(bool val)
	{
		m_ItemMoved = val;
	}

	void setPickingFinished(bool value)
	{
		m_PickingFinshed=value;
	}

	void setReadOnly(bool value)
	{
		m_readOnly = value;
	}

	void setMenu(QMenu *value)
	{
		m_Menu = value;
	}

	virtual void ContextualMenu(QPoint) = 0;
	virtual void setGrabbersVisibility(bool) =0;



//	virtual void setResizeWidth(int ,int)=0;

protected:

	QPen m_Pen;
	QMenu *m_Menu;
	bool m_AlreadySelected = true;
	bool m_ItemMoved =false;
	bool m_IsHighlighted = false;
	bool m_ItemGeometryChanged = false;
	bool m_AntialiasingEnabled = true;
	bool m_PickingFinshed = false;
	QString m_LayerID;
	QGraphicsSimpleTextItem *m_textItem = nullptr;
	bool m_readOnly = false;
};



#endif /* SRC_GRAPHICEDITOR_GraphEditor_Item_H_ */

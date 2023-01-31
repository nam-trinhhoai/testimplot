/*
 * GraphEditor_ItemInfo.h
 *
 *  Created on: Nov 2, 2021
 *      Author: l1046262
 */

#ifndef SRC_GRAPHICEDITOR_GRAPHEDITOR_ITEMINFO_H_
#define SRC_GRAPHICEDITOR_GRAPHEDITOR_ITEMINFO_H_


#include <QVector>
#include <QPointF>
#include <QList>
#include "GraphicSceneEditor.h"
#include "idata.h"


class RandomLineView;

class GraphEditor_ItemInfo
{
public:

	virtual QVector<QPointF> SceneCordinatesPoints()=0;
	virtual QVector<QPointF> ImageCordinatesPoints()=0;

	QList<IData *> WellHeadList()
	{
		if (m_scene)
		{
			if (m_scene->innerView())
			{
				return m_scene->innerView()->detectWellsIncludedInItem(dynamic_cast<QGraphicsItem *>(this));
			}
		}
		return QList<IData *>();
	}

	virtual void setRandomView(RandomLineView *pRandView)
	{
		m_View = pRandView;
	}

	RandomLineView* getRandomView(void)
	{
		return m_View;
	}

protected:
	GraphicSceneEditor* m_scene = nullptr;
	RandomLineView *m_View =nullptr;
};


#endif /* SRC_GRAPHICEDITOR_GRAPHEDITOR_ITEMINFO_H_ */

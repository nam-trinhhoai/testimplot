#ifndef IGraphicToolDataControl_H
#define IGraphicToolDataControl_H

#include <QRectF>
#include <QVector>
#include <QPointF>

#include "cudaimagepaletteholder.h"
#include "GraphEditor_RectShape.h"
#include "GraphEditor_EllipseShape.h"
#include "GraphEditor_PolygonShape.h"

class QGraphicsItem;

class iGraphicToolDataControl {
public:
	iGraphicToolDataControl() {
	}
	virtual ~iGraphicToolDataControl() {
	}

	virtual void deleteGraphicItemDataContent(QGraphicsItem *) = 0;

	template <typename T>
	static void deleteData(T *image, QGraphicsItem *item, double eraseValue = 0)
	{
		QVector<QPointF> vec = dynamic_cast<GraphEditor_ItemInfo* >(item)->SceneCordinatesPoints();
		QPolygonF item_scenePoints(vec);
		QPolygonF image_scenePoints(image->worldExtent());
		QPolygonF intersectionPoints = item_scenePoints.intersected(image_scenePoints);
		QPolygonF poly;

		foreach(QPointF p, intersectionPoints)
		{
			double i,j;
			image->worldToImage(p.x(), p.y(),i,j);
			poly.push_back(QPointF(i,j));
		}

		QPainterPath path;
		path.addPolygon(poly);

		for (int i=0; i<image->height(); i+=1)
		{
			for (int j=0; j<image->width(); j+=1)
			{
				QPointF point(j,i);
				if (path.contains(point) )
				{
					image->setValue(j,i,eraseValue);
				}
			}
		}

		emit image->dataChanged();
	}

	static std::vector<QPoint> getDeletePointsOnGrid(const IGeorefGrid* grid, QGraphicsItem *item)
	{
		QVector<QPointF> vec = dynamic_cast<GraphEditor_ItemInfo* >(item)->SceneCordinatesPoints();
		QPolygonF item_scenePoints(vec);
		QPolygonF image_scenePoints(grid->worldExtent());
		QPolygonF intersectionPoints = item_scenePoints.intersected(image_scenePoints);
		QPolygonF poly;

		foreach(QPointF p, intersectionPoints)
		{
			double i,j;
			grid->worldToImage(p.x(), p.y(),i,j);
			poly.push_back(QPointF(i,j));
		}

		QPainterPath path;
		path.addPolygon(poly);

		std::vector<QPoint> points;
		for (int i=0; i<grid->height(); i+=1)
		{
			for (int j=0; j<grid->width(); j+=1)
			{
				QPointF point(j,i);
				if (path.contains(point) )
				{
					points.push_back(QPoint(j, i));
				}
			}
		}

		return points;
	}


};

#endif

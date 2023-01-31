#ifndef GEOMETRY2DTOOLBOX_H_
#define GEOMETRY2DTOOLBOX_H_

#include <utility>
#include <QPointF>
#include <QVector2D>

#define GEOM2D_EPSILON 1.0e-30

std::pair<double, QPointF> getPointSignedProjectionOnLine(QPointF oriPoint, std::pair<QPointF, QPointF> segmentForLine, bool* ok);
std::pair<double, QPointF> getPointProjectionOnLine(QPointF oriPoint, std::pair<QPointF, QPointF> segmentForLine, bool* ok);
std::pair<double, QPointF> getPointSignedProjectionOnSegment(QPointF oriPoint, std::pair<QPointF, QPointF> segment, bool* ok);
std::pair<double, QPointF> getPointProjectionOnSegment(QPointF oriPoint, std::pair<QPointF, QPointF> segment, bool* ok);
std::tuple<double, QPointF, double, QPointF, bool> getProjectedSegmentOnSegment(std::pair<QPointF, QPointF> segmentOri,
			std::pair<QPointF, QPointF> segmentProjection, double distanceProjectionMax);
std::pair<QPointF, QVector2D> segmentToLine(std::pair<QPointF, QPointF> segment);
QVector2D getNormal(QVector2D direction, bool normalize=false);
double getSignedDistanceFromLine(QPointF point, std::pair<QPointF, QVector2D> line);
bool isOrthogonal(QVector2D line1, QVector2D line2);
bool isParallel(QVector2D line1, QVector2D line2);

#endif

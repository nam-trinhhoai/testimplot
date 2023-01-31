#ifndef PolygonInterpolator_H_
#define PolygonInterpolator_H_

#include <vector>
#include <utility>

#include <QVector3D>

#include "wellbore.h"

std::pair<Deviations, std::vector<std::size_t>> polygonInterpolator(
		const Deviations& deviation, double thresholdDeviation);
double segmentLength(QVector3D p1, QVector3D p2) ;
double scalarProduct(QVector3D p0,QVector3D p1,QVector3D p2) ;
double distPointSegment (QVector3D p0, QVector3D p1,QVector3D p2) ;

#endif

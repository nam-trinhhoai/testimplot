#include "geometry2dtoolbox.h"

#include <cmath>
#include <QDebug>

std::pair<double, QPointF> getPointSignedProjectionOnLine(QPointF oriPoint, std::pair<QPointF, QPointF> segment, bool* ok) {
	QPointF projection;
	double distance;

	QVector2D segmentAO(oriPoint - segment.first);
	QVector2D segmentAB(segment.second - segment.first);
	double lengthAB2 = segmentAB.lengthSquared();
	*ok = lengthAB2>0;
	if (*ok) {
		double t = QVector2D::dotProduct(segmentAO, segmentAB) / lengthAB2;

		projection = segment.first + segmentAB.toPointF() * t;
		distance = getSignedDistanceFromLine(oriPoint, segmentToLine(segment));//QVector2D(oriPoint).distanceToPoint(QVector2D(projection));
	} else {
		distance = std::numeric_limits<double>::max();
	}


	return std::pair<double, QPointF>(distance, projection);
}

std::pair<double, QPointF> getPointProjectionOnLine(QPointF oriPoint, std::pair<QPointF, QPointF> segment, bool* ok) {
	std::pair<double, QPointF> out = getPointSignedProjectionOnLine(oriPoint, segment, ok);
	if (out.first<0) {
		return std::pair<double, QPointF>(-out.first, out.second);
	} else {
		return out;
	}
}

// limit projection to be inside the segment
std::pair<double, QPointF> getPointSignedProjectionOnSegment(QPointF oriPoint, std::pair<QPointF, QPointF> segment, bool* ok) {
	QPointF projection;
	double distance;

	QVector2D segmentAO(oriPoint - segment.first);
	QVector2D segmentAB(segment.second - segment.first);
	double lengthAB2 = segmentAB.lengthSquared();
	*ok = lengthAB2>0;
	if (*ok) {
		double t = QVector2D::dotProduct(segmentAO, segmentAB) / lengthAB2;
		*ok = t>=0 && t<=1;

		if (*ok) {
			projection = segment.first + segmentAB.toPointF() * t;
			distance = getSignedDistanceFromLine(oriPoint, segmentToLine(segment));//QVector2D(oriPoint).distanceToPoint(QVector2D(projection));
		} else {
			distance = std::numeric_limits<double>::max();
		}
	} else {
		distance = std::numeric_limits<double>::max();
	}

	return std::pair<double, QPointF>(distance, projection);
}

std::pair<double, QPointF> getPointProjectionOnSegment(QPointF oriPoint, std::pair<QPointF, QPointF> segment, bool* ok) {
	std::pair<double, QPointF> out = getPointSignedProjectionOnSegment(oriPoint, segment, ok);
	if (out.first<0) {
		return std::pair<double, QPointF>(-out.first, out.second);
	} else {
		return out;
	}
}

std::tuple<double, QPointF, double, QPointF, bool> getProjectedSegmentOnSegment(std::pair<QPointF, QPointF> segmentOri,
			std::pair<QPointF, QPointF> segmentProjection, double distanceProjectionMax) {
	double t1, t2;
	QPointF H1, H2;
	bool ok;

	ok = (segmentOri.first - segmentOri.second).manhattanLength()!=0 && (segmentProjection.first - segmentProjection.second).manhattanLength()!=0;

	// crop segmentOri with segmentProjection limits
	// create line (Ad) orthogonal to segmentProjection with segmentProjection.first as intersection (point A)
	std::pair<QPointF, QVector2D> lineAB = segmentToLine(segmentProjection);
	QVector2D orthogonalAB = getNormal(lineAB.second);
	std::pair<QPointF, QVector2D> lineAd = std::pair<QPointF, QVector2D>(segmentProjection.first, orthogonalAB);

	// create line (Bd) orthogonal to segmentProjection with segmentProjection.first as intersection (point B)
	std::pair<QPointF, QVector2D> lineBd = std::pair<QPointF, QVector2D>(segmentProjection.second, orthogonalAB);

	// if (Ad) // (CD) check if C projection is in [AB]
	std::pair<QPointF, QVector2D> lineCD = segmentToLine(segmentOri);
	if (isOrthogonal(lineCD.second, orthogonalAB)) {
		getPointSignedProjectionOnSegment(segmentOri.first, segmentProjection, &ok);
	}

	// find intersection between lines (Ad), (Bd) and segment CD
	std::pair<QPointF, QPointF> segmentCropped;
	if (ok) {
		// crop segmentOri with segment limit of [AB]
		// use A* where A* projection on [AB] is A, same for B
		// use A* and B* to crop [CD]
	}
	segmentCropped = segmentOri; // define [Cp Dp]
	if (ok) {
		bool tmp;
		std::pair<double, QPointF> projCp = getPointSignedProjectionOnSegment(segmentCropped.first, segmentProjection, &tmp);
		std::pair<double, QPointF> projDp = getPointSignedProjectionOnSegment(segmentCropped.second, segmentProjection, &tmp);

		H1 = projCp.second;
		H2 = projDp.second;

		if (std::abs(segmentOri.second.x()-segmentOri.first.x())<GEOM2D_EPSILON) {
			t1 = (segmentCropped.first.x() - segmentOri.first.x()) / (segmentOri.second.x()-segmentOri.first.x());
			t2 = (segmentCropped.second.x() - segmentOri.first.x()) / (segmentOri.second.x()-segmentOri.first.x());
		} else {
			t1 = (segmentCropped.first.y() - segmentOri.first.y()) / (segmentOri.second.y()-segmentOri.first.y());
			t2 = (segmentCropped.second.y() - segmentOri.first.y()) / (segmentOri.second.y()-segmentOri.first.y());
		}
	}

	return std::tuple<double, QPointF, double, QPointF, bool>(t1, H1, t2, H2, ok);
}

std::pair<QPointF, QVector2D> segmentToLine(std::pair<QPointF, QPointF> segment) {
	QPointF origin = segment.first;
	QVector2D direction = QVector2D(segment.second - segment.first);
	direction.normalize();
	return std::pair<QPointF, QVector2D>(origin, direction);
}

QVector2D getNormal(QVector2D direction, bool normalize) {
	QVector2D out;
	if (direction.x()==0.0 && direction.y()==0.0) {
		out = QVector2D(0, 0);
	} else if (direction.x()==0.0) {
		out = QVector2D(1, 0);
	} else if (direction.y()==0.0) {
		out = QVector2D(0, 1);
	} else {
		if (direction.y()>0) {
			out = QVector2D(direction.y(), -direction.x());
		} else {
			out = QVector2D(-direction.y(), direction.x());
		}
		if (normalize) {
			out.normalize();
		}
	}
	return out;
}

double getSignedDistanceFromLine(QPointF point, std::pair<QPointF, QVector2D> line) {
	double a, b, c, d;
	if (line.second.x()!=0) {
		a = line.second.y();
		b = -line.second.x();
		c = -b * line.first.y() - a * line.first.x();
		d = a*point.x() + b*point.y()+c;
		d /= line.second.length();
	} else {
		a = 1;
		b = 0;
		c = -line.first.x();
		d = a*point.x() + c;
	}
	return d;
}

bool isOrthogonal(QVector2D line1, QVector2D line2) {
	return QVector2D::dotProduct(line1.normalized(), line2.normalized())<GEOM2D_EPSILON;
}

bool isParallel(QVector2D line1, QVector2D line2) {
	return ((line1.normalized() - line2.normalized()).length() < GEOM2D_EPSILON) ||
			((line1.normalized() + line2.normalized()).length() < GEOM2D_EPSILON);
}

//double getIntersectionBetweenLineAndSegment(std::pair<QPointF, QVector2D> line, std::pair<QPointF, QPointF> segment, bool* ok) {
//	double t; // relative to the segment
//
//	// reject if //
//	*ok = !isParallel(line.second, QVector2D(segment.second-segment.first));
//	if (*ok) {
//
//	}
//	return t;
//}

#include "fixedlayerfromdatasetlayeronrandom.h"

#include <QGraphicsScene>
#include <QList>
#include "fixedlayerfromdatasetreponrandom.h"
#include "cudaimagepaletteholder.h"
#include "fixedlayerfromdataset.h"
#include "curve.h"
#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "randomlineview.h"
#include "interpolation.h"

#include <cmath>

FixedLayerFromDatasetLayerOnRandom::FixedLayerFromDatasetLayerOnRandom(FixedLayerFromDatasetRepOnRandom *rep,
		QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth){
	m_rep = rep;

	m_curve = new Curve(m_scene, parent);
	QPen pen = m_curve->getPen();
	pen.setColor(Qt::red);
	m_curve->setPen(pen);
	m_curve->setZValue(m_defaultZDepth);

	// apply transform
	double tdeb = m_rep->fixedLayer()->dataset()->sampleTransformation()->b();
	double pasech = m_rep->fixedLayer()->dataset()->sampleTransformation()->a();
	QTransform transform;
	transform.translate(0, tdeb);
	transform.scale(1, pasech);
	m_curve->setTransform(transform);
	refresh();
	show();
	recomputeCurve();

	connect(m_rep->fixedLayer(), &FixedLayerFromDataset::propertyModified, this, &FixedLayerFromDatasetLayerOnRandom::recomputeCurve);
}

FixedLayerFromDatasetLayerOnRandom::~FixedLayerFromDatasetLayerOnRandom() {
	delete m_curve;
}
/*
void FixedLayerFromDatasetLayerOnSlice::setSliceIJPosition(int imageVal) {
	m_slicePosition = imageVal;
	QList<QPolygon> polygons;
	CUDAImagePaletteHolder* holder = m_rep->fixedLayer()->image(FixedLayerFromDataset::ISOCHRONE);
	if (holder) {
		std::size_t dimJ = m_rep->fixedLayer()->dataset()->height();
		std::size_t holderWidth = holder->width();
		float noValue = -9999.0;
		float pasech = m_rep->fixedLayer()->getStepSample();
		float tdeb = m_rep->fixedLayer()->getOriginSample();

		std::size_t dimI;
		if (SliceDirection::Inline==m_dir) {
			dimI = m_rep->fixedLayer()->dataset()->width();
		} else {
			dimI = m_rep->fixedLayer()->dataset()->depth();
		}

		holder->lockPointer();
		float* tab = static_cast<float*>(holder->backingPointer());
		QPolygon currentPolygon;
		for (std::size_t i=0; i<dimI; i++) {
			float val;
			if (m_dir==SliceDirection::Inline) {
				val = tab[holderWidth*m_slicePosition+i];
			} else {
				val = tab[holderWidth*i+m_slicePosition];
			}
			if (val!=noValue) {
				currentPolygon << QPoint(i, (val-tdeb)/pasech);
			} else if (val==noValue && currentPolygon.size()>0) {
				polygons << currentPolygon;
				currentPolygon = QPolygon();
			}
		}
		holder->unlockPointer();
		if (currentPolygon.size()>0) {
			polygons << currentPolygon;
		}

	}
	m_curve->setPolygons(polygons);
}*/

void FixedLayerFromDatasetLayerOnRandom::recomputeCurve() {
	QList<QPolygon> polyList;
	QPolygon poly;
	double tdeb = m_rep->fixedLayer()->dataset()->sampleTransformation()->b();
	double pasech = m_rep->fixedLayer()->dataset()->sampleTransformation()->a();
	CUDAImagePaletteHolder* isoSurface = m_rep->fixedLayer()->image(FixedLayerFromDataset::ISOCHRONE);
	if (isoSurface==nullptr) {
		return;
	}

	long width = isoSurface->width();
	long height = isoSurface->height();
	isoSurface->lockPointer();
	float* tab = (float*) isoSurface->backingPointer();

	QPolygonF polyLine = dynamic_cast<RandomLineView*>(m_rep->view())->worldDiscreatePolyLine();
	//QPolygon polyLine2 = dynamic_cast<RandomLineView*>(m_rep->view())->discreatePolyLine();
	for (long k=0; k<polyLine.size(); k++) {
		double val = 0;

		QPointF worldPoint = polyLine[k];
		double i, j;
		m_rep->fixedLayer()->dataset()->ijToXYTransfo()->worldToImage(worldPoint.x(),
				worldPoint.y(), i, j);

		//int i2 = polyLine2[k].x(), j2 = polyLine2[k].y();
		/*if (std::fabs(i2-i)>1E-9 || std::fabs(j2-j)>1E-9) {
			qDebug() << ".?.?.?.";
		}*/

		std::vector<BilinearPoint> interpPoints = bilinearInterpolationPoints(i, j, 0.0,
				0.0, width-1, height-1, 1.0, 1.0);

		/*if (interpPoints.size()==1 && (std::fabs(i2-interpPoints[0].i)>1E-9 || std::fabs(j2-interpPoints[0].j)>1E-9)) {
			qDebug() << ".?.?.?.";
		}*/

		double w = 0;
		for (BilinearPoint pt : interpPoints) {
			float valTab = tab[((long)pt.i)+((long)pt.j)*width];
			if (valTab!=-9999.0) {
				val += pt.w * valTab;
				w += pt.w;
			}
		}
		if (w!=0.0) {
			val /= w;

			val = (val - tdeb) / pasech;
			if (interpPoints.size()>0) {
				poly << QPoint(k, val);
			}
		} else if (poly.size()>0) {
			polyList << poly;
			poly = QPolygon();
		}
	}
	isoSurface->unlockPointer();

	if (poly.size()>0) {
		polyList << poly;
	}

	m_curve->setPolygons(polyList);
}

void FixedLayerFromDatasetLayerOnRandom::show() {
	m_isShown = true;
	m_curve->addToScene();
}
void FixedLayerFromDatasetLayerOnRandom::hide() {
	m_isShown = false;
	m_curve->removeFromScene();
}

QRectF FixedLayerFromDatasetLayerOnRandom::boundingRect() const {
	return m_curve->boundingRect();
}

void FixedLayerFromDatasetLayerOnRandom::refresh() {
	m_curve->redraw();
	if (m_isShown) {
		show();
	} else {
		hide();
	}
}

void FixedLayerFromDatasetLayerOnRandom::setPenColor(QColor color) {
	QPen pen = m_curve->getPen();
	pen.setColor(color);
	m_curve->setPen(pen);
}

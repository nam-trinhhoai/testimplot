#include "wellpolylinemanager.h"
#include "MuratCanvas2dScene.h"

#include <QDebug>
#include <QPainterPath>
#include <QGraphicsPathItem>
#include <QVector2D>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QDoubleValidator>

#include <math.h>

QPointF getNormalVector(QPointF vect) {
    QPointF out;
    if (vect.x()==0 && vect.y()!=0) {
        out = QPointF(1, 0);
    } else if (vect.x()!=0 && vect.y()==0) {
        out = QPointF(0, 1);
    } else if (vect.x()!=0 && vect.y()!=0) {
    	double b1, b2;
    	if (vect.x()>vect.y()) {
    		b2 = fabs(vect.x()/sqrt(pow(vect.x(), 2)+pow(vect.y(), 2)));
    		b1 = - b2 * vect.y() / vect.x();
    	} else {
    		b1 = fabs(vect.y()/sqrt(pow(vect.y(), 2)+pow(vect.x(), 2)));
    		b2 = - b1 * vect.x() / vect.y();
    	}
        out = QPointF(b1, b2);
    }
    double n2 = pow(out.x(), 2) + pow(out.y(), 2);
    if (n2<0.999 || n2>1.001) {
        qDebug() << "getNormalVector with vect:" << vect << " give out:" << out << " and ||out||²" << n2;
    }
    return out;
}

void moy1d_double(QPolygonF& tab1,QPolygonF& tab2,int nx,int fx,int opt)
{
        int i,j,ix,ind;
        double som ;
        tab2[0] = QPointF(0,0);
        for (ix=0;ix<=fx;ix++) { tab2[0] = tab2[0] +  tab1[ix];}
        for (ix=1;ix<=fx;ix++)
               tab2[ix]=tab2[ix -1]+tab1[ix+fx];
               /* Partie centrale */
        for (ix=fx+1;ix<nx-fx;ix++)
                tab2[ix]= tab2[ix -1]+ tab1[ ix + fx]- tab1[ix-fx-1];

                /* Conditions finales */
        for (ix=nx-fx;ix<nx;ix++)
                tab2[ix]=tab2[ix-1]-tab1[ix-fx-1];
               /* Renormalisations */
        if(opt != 0) {
                for (ix=0;ix<=fx;ix++) tab2[ix]/=(ix+fx+1);
                for (ix=fx+1;ix<nx-fx;ix++) tab2[ix] /= (2*fx+1);
                for (ix=nx-fx;ix<nx;ix++) tab2[ix]/=(nx-ix+fx);
        }
}


WellPolylineManager::WellPolylineManager(QObject *parent) : QObject(parent)
{

}

WellPolylineManager::~WellPolylineManager() {
    for(int i=viewers.size()-1; i>=0; i--) {
        disconnectViewer(viewers[i].view);
    }
}

void WellPolylineManager::connectViewer(MuratCanvas2dFullView* viewer) {
    if (viewer != nullptr) {
        PolylineViewer polyViewer;
        polyViewer.view = viewer;

        if (viewers.size()==0) {
            cumul_factorX = viewer->getRatioX();
            cumul_factorY = viewer->getRatioY();
        }

        //if (!well.name.isNull() && !well.name.isEmpty() && well.samples.size()>0) {
            polyViewer.supports = new QGraphicsPathItem;
            polyViewer.only_supports = new QGraphicsPathItem;
            polyViewer.values = new QGraphicsPathItem;

            QPainterPath path;
            for (int i=0; i<supports.size(); i++) {
                if (wells[i].samples.size()>0) {
                    const QVector<QPolygonF>& well_supports = supports[i];
                    for(const QPolygonF& poly : well_supports) {
                        path.addPolygon(poly);
                    }
                }
            }
            polyViewer.supports->setPath(path);

            path = QPainterPath();
            for (int i=0; i<supports.size(); i++) {
                if (wells[i].samples.size()==0) {
                    const QVector<QPolygonF>& well_supports = supports[i];
                    for(const QPolygonF& poly : well_supports) {
                        path.addPolygon(poly);
                    }
                }
            }
            polyViewer.only_supports->setPath(path);

            path = QPainterPath();
            for (const QVector<QPolygonF>& well_values: values) {
                for(const QPolygonF& poly : well_values) {
                    path.addPolygon(poly);
                }
            }
            polyViewer.values->setPath(path);

            MuratCanvas2dScene* canvas = viewer->getCanvas();
            canvas->addItem(polyViewer.supports);
            canvas->addItem(polyViewer.only_supports);
            canvas->addItem(polyViewer.values);
            polyViewer.supports->setZValue(10);
            polyViewer.only_supports->setZValue(10);
            polyViewer.values->setZValue(10);
        //}

        polyViewer.connection = std::make_shared<QMetaObject::Connection>();
        *(polyViewer.connection) = connect(viewer, &MuratCanvas2dFullView::zoomed, this, [this](double factorX, double factorY, QPointF target_scene_pos) {
            this->changeScale(factorX, factorY);
        });
        viewers.append(polyViewer);

        initPens(polyViewer);
    }
}

void WellPolylineManager::disconnectViewer(MuratCanvas2dFullView* viewer) {
    // Search in that order to optimize destructor
    int index = viewers.size()-1;
    while (index>=0 && viewers[index].view!=viewer) {
        index--;
    }
    if (index>=0) {
        PolylineViewer polyViewer = viewers[index];
        viewers.remove(index);

        if (polyViewer.supports!=nullptr) {
            viewer->getCanvas()->removeItem(polyViewer.supports);
            delete polyViewer.supports;
        }
        if (polyViewer.values!=nullptr) {
            viewer->getCanvas()->removeItem(polyViewer.values);
            delete polyViewer.values;
        }

        disconnect(*polyViewer.connection);
    }
}

void WellPolylineManager::setOrientation(SectionOrientation orientation) {
    if (this->orientation != orientation) {
        if (this->orientation == SectionOrientation::RANDOM) {
            randomPts.clear();
        }

        this->orientation = orientation;

        updateViewers();
    }
}

void WellPolylineManager::setRandomLine(const QVector<QPoint>& random_line) {
    randomPts = random_line;
    if (orientation == SectionOrientation::RANDOM) {
        updateViewers();
    }
}

void WellPolylineManager::addWell(const Well& well, QVector<QVector3D> wholeTrajectory) {
    if (well.name.isEmpty() || well.name.isNull()) {
        return;
    }
    this->wells.append(well);
    this->wholeTrajectories.append(wholeTrajectory);
    supports.append(QVector<QPolygonF>());
    values.append(QVector<QPolygonF>());
    updateViewers();
}

void WellPolylineManager::removeWell(QString well_name) {
    int index = 0;
    while (index<wells.size() && wells[index].name.compare(well_name)!=0) {
        index ++;
    }
    if (index<wells.size()) {
        wells.remove(index);
        wholeTrajectories.remove(index);
        supports.remove(index);
        values.remove(index);
    }
    updateViewers();
}

bool WellPolylineManager::containsWell(const QString& name) {
    int index = 0;
    while (index<wells.size() && wells[index].name.compare(name)!=0) {
        index ++;
    }
    return index<wells.size();
}

void WellPolylineManager::setSlice(long slice) {
    this->slice = slice;
    if (orientation==SectionOrientation::INLINE || orientation==SectionOrientation::XLINE) {
        updateViewers();
    }
}

void WellPolylineManager::updateViewers() {
    clearDisplay();

    if (orientation==SectionOrientation::RANDOM && randomPts.size()==0) {
        return;
    }

    computeSupportsAndValues();

    for (PolylineViewer& polyViewer : viewers) {
        if (polyViewer.supports==nullptr) {
            polyViewer.supports = new QGraphicsPathItem;
            polyViewer.view->getCanvas()->addItem(polyViewer.supports);
            polyViewer.supports->setZValue(10);
        }
        if (polyViewer.only_supports==nullptr) {
            polyViewer.only_supports = new QGraphicsPathItem;
            polyViewer.view->getCanvas()->addItem(polyViewer.only_supports);
            polyViewer.only_supports->setZValue(10);
        }
        if (polyViewer.values==nullptr) {
            polyViewer.values = new QGraphicsPathItem;
            polyViewer.view->getCanvas()->addItem(polyViewer.values);
            polyViewer.values->setZValue(10);
        }

        QPainterPath path;
        QPainterPath path_only_support;
        for (int i=0; i<supports.size(); i++) {
            const QVector<QPolygonF>& well_supports = supports[i];
            if (wells[i].samples.size()>0) {
                for (const QPolygonF& support : well_supports) {
                    path.addPolygon(support);
                }
            } else {
                for (const QPolygonF& support : well_supports) {
                    path_only_support.addPolygon(support);
                }
            }
        }
        polyViewer.supports->setPath(path);
        polyViewer.only_supports->setPath(path_only_support);

        path = QPainterPath();
        for (const QVector<QPolygonF>& well_values: values) {
            for (const QPolygonF& value : well_values){
                path.addPolygon(value);
            }
        }
        polyViewer.values->setPath(path);

        initPens(polyViewer);
    }

}

void WellPolylineManager::computeSupportsAndValues() {
    for (int i=0; i<supports.size(); i++) {
        supports[i].clear();
    }
    for (int i=0; i<values.size(); i++) {
        values[i].clear();
    }

    for (int wc=0; wc<supports.size(); wc++) {
        QVector<int> firstIndexes;
        supports[wc].append(QPolygonF());
        for (int j=0; j<wholeTrajectories[wc].size(); j++) {
            double py = wholeTrajectories[wc][j].y();
            double pz = wholeTrajectories[wc][j].x();

            double y_section = -1;
            double dist;
            if(orientation==SectionOrientation::INLINE) {
                dist = fabs(pz - slice);
                y_section = py;
            } else if(orientation==SectionOrientation::XLINE) {
                dist = fabs(py - slice);
                y_section = pz;
            } else { // orientation == random
                dist = std::numeric_limits<double>::max();
                for (int index=0; index<randomPts.size(); index++) {
                    QPoint& pt = randomPts[index];
                    double val = sqrt(pow(py - pt.y(), 2) + pow(pz - pt.x(), 2));
                    if (dist>val) {
                        dist = val;
                        y_section = index;
                    }
                    dist = std::min(dist, val);
                }
            }
            if (dist<this->well_projection_max_distance) {
                QPolygonF& polygon_support = supports[wc].last();
                QPointF target_point(y_section, wholeTrajectories[wc][j].z()*ratioBetweenWellsAndImage);
                polygon_support.append(target_point);

                if (polygon_support.size()==1) {
                    firstIndexes.append(j);
                }

                /*QPolygonF& polygon_value = values.last();
                if (polygon_support.size()>1 && well.samples.size()>0) {
                    long N = polygon_support.size();
                    QPointF a = polygon_support[N-1];
                    QPointF b = polygon_support[N-2];
                    QPointF delta = a - b;
                    QPointF normal = getNormalVector(delta);
                    QPointF res = a + normal / cumul_factor * (offset_min + size_min_max * (well.samples[j].logVals[0] - logDynamic[0].first) / (logDynamic[0].second - logDynamic[0].first));

                    if (polygon_support.size()==2) {
                        QPointF first_res = b + normal / cumul_factor * (offset_min + size_min_max * (well.samples[j-1].logVals[0] - logDynamic[0].first) / (logDynamic[0].second - logDynamic[0].first));
                        polygon_value.append(first_res);
                    }
                    polygon_value.append(res);
                }*/
            } else {
                if(supports[wc].last().size()!=0) {
                    supports[wc].append(QPolygonF());
                }
                /*if(values.last().size()!=0) {
                    values.append(QPolygonF());
                }*/
                qDebug() << "Ignore point" << wholeTrajectories[wc][j].z() << py << pz;
            }
        }
        if (supports[wc].last().size()==0) {
            supports[wc].removeLast();
        }
        // remove polygons that do not have enough points for filtering
        for (int i_destruct = supports[wc].size()-1; i_destruct>=0; i_destruct--) {
        	if(supports[wc][i_destruct].size()<2) {
        		supports[wc].removeAt(i_destruct);
        	}
        }
        for (int i=0; i<supports[wc].size(); i++) {
            QPolygonF tmp_poly(supports[wc][i]);
            moy1d_double(supports[wc][i], tmp_poly, supports[wc][i].size(), 2, 1);
            supports[wc][i] = tmp_poly;
        }

        for (int k=0; k<supports[wc].size(); k++) {
            QPolygonF polygon_support = supports[wc][k];
            QPolygonF polygon_value;
            for (int i=1; i<polygon_support.size(); i++) {
                int j = i + firstIndexes[k];
                if (polygon_support.size()>1 && wells[wc].samples.size()>0) {
                    QPointF a = polygon_support[i];
                    QPointF b = polygon_support[i-1];
                    QPointF delta = a - b;
                    QPointF normal = getNormalVector(delta);

                    float normalFactor = ((float) offset_min) + ((float) size_min_max) * (wells[wc].samples[j-1].logVals[valueIndex] - logDynamic[valueIndex].first) / (logDynamic[valueIndex].second - logDynamic[valueIndex].first);

                    qDebug() << normalFactor << offset_min << size_min_max << wells[wc].samples[j-1].logVals[valueIndex] << logDynamic[valueIndex].first << logDynamic[valueIndex].second << (wells[wc].samples[j-1].logVals[valueIndex] - logDynamic[valueIndex].first) << logDynamic[valueIndex].second - logDynamic[valueIndex].first;

                    QPointF res = a + normal / getCumulFactor() * normalFactor;

                    if (i==1) {
                        QPointF first_res = b + normal / getCumulFactor() * normalFactor;
                        polygon_value.append(first_res);
                    }
                    polygon_value.append(res);
                }
            }
            values[wc].append(polygon_value);
        }
    }
}

void WellPolylineManager::clearDisplay() {
    for (PolylineViewer& polyViewer : viewers) {
        if (polyViewer.supports!=nullptr) {
            QPainterPath path;
            polyViewer.supports->setPath(path);
        }
        if (polyViewer.values!=nullptr) {
            QPainterPath path;
            polyViewer.values->setPath(path);
        }
    }
}

long WellPolylineManager::getWellProjectionMaxDistance() {
    return well_projection_max_distance;
}

void WellPolylineManager::setWellProjectionMaxDistance(long val) {
    well_projection_max_distance = val;
}

void WellPolylineManager::setLogDynamic(QVector<std::pair<float, float> > logDynamic) {
    this->logDynamic = logDynamic;
}

void WellPolylineManager::changeScale(double factorX, double factorY) {
    int index = 0;
    cumul_factorX = cumul_factorX * factorX;
    cumul_factorY = cumul_factorY * factorY;
    /*while (index<viewers.size()) {
        PolylineViewer& polyViewer = viewers[index];
        QPen support_pen = polyViewer.supports->pen();
        support_pen.setWidthF(pen_width / cumul_factor);
        polyViewer.supports->setPen(support_pen);

        QPen value_pen = polyViewer.values->pen();
        value_pen.setWidthF(pen_width / cumul_factor);
        polyViewer.values->setPen(value_pen);
        index ++;
    }*/

    updateViewers();
}


double WellPolylineManager::getCumulFactor() {
    return cumul_factorX;
}

void WellPolylineManager::runGraphicsSettingsDialog(QWidget* dialogParent) {
	WellPolylineManagerSettingsDialog dialog(pen_width, pen_width_support_only, size_min_max, offset_min, dialogParent);

	int code = dialog.exec();
	if (code == QDialog::Accepted) {
		pen_width = dialog.getPenWidth();
		pen_width_support_only = dialog.getPenWidthSupportOnly();
		size_min_max = dialog.getSizeMinMax();
		offset_min = dialog.getOffsetMin();

		// redraw
		updateViewers();
	}

}

void WellPolylineManager::setValueIndex(int valueIndex) {
	this->valueIndex = valueIndex;
}

int WellPolylineManager::getValueIndex() {
	return valueIndex;
}

void WellPolylineManager::setRatioBetweenWellsAndImage(double newRatio) {
	if (ratioBetweenWellsAndImage!=newRatio) {
		ratioBetweenWellsAndImage = newRatio;
		updateViewers();
	}
}

void WellPolylineManager::initPens(PolylineViewer polyViewer) {
    QPen supportPen(color_full_well_support, pen_width);
    supportPen.setCosmetic(true);
    polyViewer.supports->setPen(supportPen);

    QPen onlySupportPen(color_support_only, pen_width_support_only);
    onlySupportPen.setCosmetic(true);
    polyViewer.only_supports->setPen(onlySupportPen);

    QPen valuesPen(color_full_well_value, pen_width);
    valuesPen.setCosmetic(true);
    polyViewer.values->setPen(valuesPen);
}

WellPolylineManagerSettingsDialog::WellPolylineManagerSettingsDialog(double pen_width, double pen_width_support_only, double size_min_max, double offset_min, QWidget* parent) : QDialog(parent) {
	this->pen_width = pen_width;
	this->pen_width_support_only = pen_width_support_only;
	this->size_min_max = size_min_max;
	this->offset_min = offset_min;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	this->setLayout(mainLayout);

	QFormLayout* form = new QFormLayout;
	mainLayout->addLayout(form);

	QLineEdit* lineEdit = new QLineEdit;
	lineEdit->setText(locale().toString(pen_width));
	QDoubleValidator* validator = new QDoubleValidator(std::numeric_limits<double>::min(),
				std::numeric_limits<double>::max(),
				std::numeric_limits<int>::max());
	lineEdit->setValidator(validator);
	connect(lineEdit, &QLineEdit::textChanged, this, [this](const QString& txt) {
		bool test;
		this->pen_width = this->locale().toFloat(txt, &test);
	});
	form->addRow("Pen Width", lineEdit);

	lineEdit = new QLineEdit;
	lineEdit->setText(locale().toString(pen_width_support_only));
	validator = new QDoubleValidator(std::numeric_limits<double>::min(),
					std::numeric_limits<double>::max(),
					std::numeric_limits<int>::max());
	lineEdit->setValidator(validator);
	connect(lineEdit, &QLineEdit::textChanged, this, [this](const QString& txt) {
		bool test;
		this->pen_width_support_only = this->locale().toFloat(txt, &test);
	});
	form->addRow("Pen Width Support Only", lineEdit);

	lineEdit = new QLineEdit;
	lineEdit->setText(locale().toString(size_min_max));
	validator = new QDoubleValidator(std::numeric_limits<double>::min(),
					std::numeric_limits<double>::max(),
					std::numeric_limits<int>::max());
	lineEdit->setValidator(validator);
	connect(lineEdit, &QLineEdit::textChanged, this, [this](const QString& txt) {
		bool test;
		this->size_min_max = this->locale().toFloat(txt, &test);
	});
	form->addRow("Size to draw logs values", lineEdit);

	lineEdit = new QLineEdit;
	lineEdit->setText(locale().toString(offset_min));
	validator = new QDoubleValidator(-std::numeric_limits<double>::max(),
					std::numeric_limits<double>::max(),
					std::numeric_limits<int>::max());
	lineEdit->setValidator(validator);
	connect(lineEdit, &QLineEdit::textChanged, this, [this](const QString& txt) {
		bool test;
		this->offset_min = this->locale().toFloat(txt, &test);
	});
	form->addRow("Offset between support and values", lineEdit);

	QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(buttonBox, &QDialogButtonBox::accepted, this, &WellPolylineManagerSettingsDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &WellPolylineManagerSettingsDialog::reject);

	mainLayout->addWidget(buttonBox);
}

WellPolylineManagerSettingsDialog::~WellPolylineManagerSettingsDialog() {}

double WellPolylineManagerSettingsDialog::getPenWidth() {
	return pen_width;
}

double WellPolylineManagerSettingsDialog::getPenWidthSupportOnly() {
	return pen_width_support_only;
}

double WellPolylineManagerSettingsDialog::getSizeMinMax() {
	return size_min_max;
}

double WellPolylineManagerSettingsDialog::getOffsetMin() {
	return offset_min;
}


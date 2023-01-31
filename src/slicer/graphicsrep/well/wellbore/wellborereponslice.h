#ifndef WellBoreRepOnSlice_H
#define WellBoreRepOnSlice_H

#include <QObject>
#include <QRectF>
#include <QVector2D>

#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "abstractgraphicrep.h"
#include "sliceutils.h"
#include "viewutils.h"

class WellBore;
class WellBoreLayerOnSlice;

class WellBoreRepOnSlice: public AbstractGraphicRep, public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
	typedef struct Trajectory {
		std::vector<double> samples, traces, profils, xs, ys, mds;
	} Trajectory;

	typedef struct LogGraphicPoint {
		QPointF refPoint;
		QVector2D normal;
		double logValue;
	} LogGraphicPoint;



	WellBoreRepOnSlice(WellBore *wellBore, SliceDirection dir = SliceDirection::Inline, 
			AbstractInnerView *parent = 0);
	virtual ~WellBoreRepOnSlice();

	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	virtual void setSliceIJPosition(int val) override; // to be aware of position change but discard val
	int currentSliceWorldPosition() const;
	SliceDirection direction() const {
		return m_dir;
	}

	const std::vector<QPolygonF>& displayTrajectories() const;
	QRectF boundingBox() const;

	virtual bool setSampleUnit(SampleUnit type) override; // m_dir need to be set before calling, needed for bounding box computation
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual void buildContextMenu(QMenu *menu) override;
	virtual TypeRep getTypeGraphicRep() override;
	bool isLayerShown() const;

	const std::vector<std::vector<LogGraphicPoint>>& logDisplayParams() const;

	double displayDistance() const;
	void setDisplayDistance(double);
signals:
    void deletedRep(AbstractGraphicRep *rep);// MZR 15072021
private slots:
	void updatePicks();
	void logChanged();
	void logMinChanged(double);
	void logMaxChanged(double);
	void widthChanged(double);
	void originChanged(double);
	void deleteWellBoreRepOnSlice();
	void viewWellsLog();
	void reExtractDeviation();

private:
	QRectF computeBoundingBox();
	void logPreprocessing(const std::vector<std::pair<double, double>>& mdIntervals);
	std::vector<std::pair<double, double>> fuseIntervals(const std::vector<std::pair<double, double>>& mdIntervals,
			const std::vector<std::pair<long, long>>& logIndexInterval);

	int m_currentWorldSlice;
	SliceDirection m_dir;

	WellBoreLayerOnSlice *m_layer;
	QWidget* m_propPanel;
	WellBore* m_data;
	SampleUnit m_sectionType = SampleUnit::NONE;
	Trajectory m_trajectory; // in scene coordinates,
	std::vector<QPolygonF> m_displayTrajectories; // trajectory projected on slice (array of subTraces)
	std::vector<QPolygonF> m_displayLogTrajectories;
	std::vector<std::vector<LogGraphicPoint>> m_logDisplayParams;
	QRectF m_boundingBox; // bbox is computed here when m_trajectory is computed

	double m_origin = 50.0;
	double m_width = 100.0;
	double m_logMin = 0.0;
	double m_logMax = 100;

	double m_displayDistance;
};

#endif

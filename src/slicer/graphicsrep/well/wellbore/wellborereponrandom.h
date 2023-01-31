#ifndef WellBoreRepOnRandom_H_
#define WellBoreRepOnRandom_H_

#include "abstractgraphicrep.h"
#include "isampledependantrep.h"
#include "affinetransformation.h"

#include <QVector2D>

class WellBore;
class WellBoreLayerOnRandom;
class GraphicLayer;
class RandomLineView;

class WellBoreRepOnRandom : public AbstractGraphicRep, public ISampleDependantRep
{
Q_OBJECT
public:
	typedef struct LogGraphicPoint {
		QPointF refPoint;
		QVector2D normal;
		double logValue;
	} LogGraphicPoint;

	WellBoreRepOnRandom(WellBore *data, AbstractInnerView *parent = 0);
	virtual ~WellBoreRepOnRandom();

	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;

	const std::vector<QPolygonF>& displayTrajectories() const;
	QRectF boundingBox() const;

	virtual bool setSampleUnit(SampleUnit type) override; // m_dir need to be set before calling, needed for bounding box computation
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual void buildContextMenu(QMenu *menu) override; // MZR 18082021
	virtual TypeRep getTypeGraphicRep() override;
	virtual void deleteLayer() override;
	bool isLayerShown() const;

	double displayDistance() const;
	void setDisplayDistance(double val);
	const std::vector<std::vector<LogGraphicPoint>>& logDisplayParams() const;

private slots:
	void updatePicks();
	void logChanged();
	void logMinChanged(double);
	void logMaxChanged(double);
	void widthChanged(double);
	void originChanged(double);
	void deleteWellBoreRepOnRandom();// MZR 19082021
	void reExtractDeviation();

signals:
    void deletedRep(AbstractGraphicRep *rep);// MZR 19082021

private:
	QRectF computeBoundingBox();
	// do not forget to clean up if return is false
	bool buildTrajectoriesOnRandom(SampleUnit type);

	void logPreprocessing(const std::vector<std::pair<double, double>>& mdIntervals, std::pair<QPointF, QPointF> randomSegment);
	std::vector<std::pair<double, double>> fuseIntervals(const std::vector<std::pair<double, double>>& mdIntervals,
			const std::vector<std::pair<long, long>>& logIndexInterval);

	QWidget* m_propPanel;
	WellBoreLayerOnRandom *m_layer;
	WellBore* m_data;
	SampleUnit m_sectionType = SampleUnit::NONE;
	std::vector<QPolygonF> m_displayTrajectories; // trajectory projected on slice (array of subTraces)
	std::vector<QPolygonF> m_displayLogTrajectories;
	std::vector<std::vector<LogGraphicPoint>> m_logDisplayParams;
	QRectF m_boundingBox; // bbox is computed here when m_trajectory is computed

	double m_displayDistance;
};

#endif

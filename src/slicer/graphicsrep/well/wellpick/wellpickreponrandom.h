#ifndef WellPickRepOnRandom_H
#define WellPickRepOnRandom_H

#include <QObject>
#include <QMap>
#include <QVector3D>

#include "isampledependantrep.h"
#include "abstractgraphicrep.h"

class WellPick;
class WellPickLayerOnRandom;
class WellBoreRepOnRandom;

class WellPickRepOnRandom: public AbstractGraphicRep, public ISampleDependantRep {
Q_OBJECT
public:
	WellPickRepOnRandom(WellPick* wellPick, AbstractInnerView *parent = 0);
	virtual ~WellPickRepOnRandom();

	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override {
		return true;
	}

	WellPick* wellPick() const;

	// x is a numero of trace, y is a numero of inline, z is sample (height on slice)
	bool isCurrentPointSet() const;
	const QList<QPointF>& getCurrentPointList(bool* ok) const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual void buildContextMenu(QMenu *menu) override; // MZR 18082021
	virtual TypeRep getTypeGraphicRep() override;
	virtual void deleteLayer() override;
	bool linkedRepShown() const;
	bool isLinkedRepValid() const;

	void searchWellBoreRep();

	double displayDistance() const;
	void setDisplayDistance(double);
signals:
	void deletedRep(AbstractGraphicRep *rep);// MZR 15072021

public slots:
	void wellBoreRepDeleted();
	void wellBoreLayerChanged(bool toggle, WellBoreRepOnRandom* originObj);
	void deleteWellPickRepOnRandom(); // 18082021
	void reExtractPosition();
private:
	void updateLayer();

	WellPickLayerOnRandom *m_layer;
	WellPick* m_data;

	bool m_isPointSet = false;
	QList<QPointF> m_points; // there can be many representation of the same WellPick, it depends of the random line

	SampleUnit m_sectionType = SampleUnit::NONE;
	WellBoreRepOnRandom* m_linkedRep = nullptr;
	bool m_linkedRepShown = false;

	double m_displayDistance;
};

#endif

#ifndef WellPickRepOnSlice_H
#define WellPickRepOnSlice_H

#include <QObject>
#include <QMap>
#include <QVector3D>

#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "abstractgraphicrep.h"

class WellPick;
class WellPickLayerOnSlice;
class WellBoreRepOnSlice;

class WellPickRepOnSlice: public AbstractGraphicRep, public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
	WellPickRepOnSlice(WellPick* wellPick, AbstractInnerView *parent = 0);
	virtual ~WellPickRepOnSlice();

	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override {
		return true;
	}

	WellPick* wellPick() const;

	// x is a numero of trace, y is a numero of inline, z is sample (height on slice)
	bool isCurrentPointSet() const;
	QVector3D getCurrentPoint(bool* ok) const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	virtual void setSliceIJPosition(int val) override;
	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual void buildContextMenu(QMenu *menu) override;
	virtual TypeRep getTypeGraphicRep() override;
	bool linkedRepShown() const;
	bool isLinkedRepValid() const;

	void searchWellBoreRep();

	double displayDistance() const;
	void setDisplayDistance(double val);

public slots:
	void wellBoreRepDeleted();
	void wellBoreLayerChanged(bool toggle, WellBoreRepOnSlice* originObj);
	void deleteWellPickRepOnSlice(); // 18082021
	void reExtractPosition();

signals:
	void deletedRep(AbstractGraphicRep *rep);// MZR 18082021

private:
	WellPickLayerOnSlice *m_layer;
	WellPick* m_data;

	bool m_isPointSet = false;
	QVector3D m_point; // x is a numero of trace, y is a numero of inline, z is sample (height on slice)

	SampleUnit m_sectionType = SampleUnit::NONE;
	WellBoreRepOnSlice* m_linkedRep = nullptr;
	bool m_linkedRepShown = false;

	double m_displayDistance;
};

#endif

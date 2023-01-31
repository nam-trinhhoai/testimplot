#ifndef WellPickRep_H
#define WellPickRep_H

#include <QObject>
#include <QMap>
#include <QVector3D>

#include "isampledependantrep.h"
#include "abstractgraphicrep.h"

class WellPick;
class WellPickLayer3D;
class WellBoreRepOn3D;

class WellPickRep: public AbstractGraphicRep, public ISampleDependantRep {
Q_OBJECT
public:
	WellPickRep(WellPick* wellPick, AbstractInnerView *parent = 0);
	virtual ~WellPickRep();

	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override {
		return true;
	}

	// x is a numero of trace, y is a numero of inline, z is sample (height on slice)
	bool isCurrentPointSet() const;
	QVector3D getCurrentPoint(bool* ok) const;

	QVector3D getDirection(SampleUnit unit ,bool* ok) const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;

	void updateLayer();
	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual TypeRep getTypeGraphicRep() override;
	bool linkedRepShown() const;
	bool isLinkedRepValid() const;

	void searchWellBoreRep();

	SampleUnit sampleUnit() const;

public slots:
	void wellBoreRepDeleted();
	void wellBoreLayerChanged(bool toggle, WellBoreRepOn3D* originObj);
	void reExtractPosition();

private:
	WellPickLayer3D *m_layer3D;
	WellPick* m_data;

	bool m_isPointSet = false;
	QVector3D m_point; // x is a numero of trace, y is a numero of inline, z is sample (height on slice)

	SampleUnit m_sectionType = SampleUnit::NONE;
	WellBoreRepOn3D* m_linkedRep = nullptr;
	bool m_linkedRepShown = false;

	QVector3D m_direction;
};

#endif

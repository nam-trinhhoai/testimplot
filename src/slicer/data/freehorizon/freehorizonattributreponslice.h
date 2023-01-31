
#ifndef __FREEHORIZONATTRIBUTREP__
#define __FREEHORIZONATTRIBUTREP__

#include <QObject>
#include <QMap>
#include <QVector3D>

#include <sliceutils.h>
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "abstractgraphicrep.h"
#include <abstractinnerview.h>
// #include "fixedlayerfromdatasetproppanel.h"

class FreeHorizonAttribut;
class FreeHorizonPropPanel;


class FreeHorizonAttributRepOnSlice: public AbstractGraphicRep{ //, public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
FreeHorizonAttributRepOnSlice(FreeHorizonAttribut* freehorizonattribut, SliceDirection dir, AbstractInnerView *parent = 0);
	virtual ~FreeHorizonAttributRepOnSlice();

	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override {
		return true;
	}

	virtual QWidget* propertyPanel() override;
	SliceDirection direction() const {return m_dir;}
	virtual void buildContextMenu(QMenu *menu) override;


	virtual TypeRep getTypeGraphicRep() override;

	// virtual void setSliceIJPosition(int val) override;



	/*
	// x is a numero of trace, y is a numero of inline, z is sample (height on slice)
	bool isCurrentPointSet() const;
	QVector3D getCurrentPoint(bool* ok) const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;


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

signals:
	void deletedRep(AbstractGraphicRep *rep);// MZR 18082021
	*/

private:
	// FreeHorizonAttributRep *m_layer;
	FreeHorizonAttribut* m_data;
	SliceDirection m_dir;
	FreeHorizonPropPanel *m_propPanel = nullptr;


	/*
	bool m_isPointSet = false;
	QVector3D m_point; // x is a numero of trace, y is a numero of inline, z is sample (height on slice)

	SampleUnit m_sectionType = SampleUnit::NONE;
	WellBoreRepOnSlice* m_linkedRep = nullptr;
	bool m_linkedRepShown = false;

	double m_displayDistance;
	*/
};

#endif

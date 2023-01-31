#ifndef WellBoreRepOn3D_H
#define WellBoreRepOn3D_H

#include <QObject>
#include <QMap>
#include <QColor>

#include "abstractgraphicrep.h"
#include "isampledependantrep.h"

class QWidget;
class WellBore;
class WellBoreLayer3D;
class WellBorePropPanelOn3D;

// 01/04/2022 Armand Sibille
// The rep/layer/proppanel need to be checked
// There is too many issues with the property panel :
// Property panel does not initialize properly, view3d settings changes do not affect the property panel
// Property panel update does not update everything
// Some property panel parameters are missing
// why use the view3D->send????() functions instead of using proper initialization ?
// why does the layer3d init function init a parameter in the property panel ?
class WellBoreRepOn3D: public AbstractGraphicRep, public ISampleDependantRep {
Q_OBJECT
public:
	WellBoreRepOn3D(WellBore *wellBore, AbstractInnerView *parent = 0);
	virtual ~WellBoreRepOn3D();

	WellBore* wellBore() const;
	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override {
		return true;
	}
	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;

	SampleUnit sampleUnit() const;
	virtual bool setSampleUnit(SampleUnit type) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual void buildContextMenu(QMenu *menu) override;
	virtual TypeRep getTypeGraphicRep() override;
	bool isLayerShown() const;

private slots:
	void updatePicks();
	void updatedParameterState();
	void deleteWellBoreRepOn3D();
	void reExtractDeviation();

signals:
	void deletedRep(AbstractGraphicRep *rep);// MZR 18082021

private:
	WellBoreLayer3D *m_layer3D;
	WellBorePropPanelOn3D* m_propPanel;
	WellBore* m_data;

	long m_defaultWidth = 40;
	long m_minimalWidth = 50;
	long m_maximalWidth = 90;
	double m_logMin = -1;
	double m_logMax = 1;
	QColor m_defaultColor = QColor(255, 255, 0);

	SampleUnit m_sampleUnit = SampleUnit::NONE;
};

#endif


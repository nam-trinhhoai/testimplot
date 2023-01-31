#ifndef FixedLayerFromDatasetRepOnRandom_H
#define FixedLayerFromDatasetRepOnRandom_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"

#include <QColor>

class FixedLayerFromDatasetPropPanelOnRandom;
class FixedLayerFromDatasetLayerOnRandom;
class FixedLayerFromDataset;
class IGeorefImage;

//For Section
class FixedLayerFromDatasetRepOnRandom: public AbstractGraphicRep, public ISampleDependantRep {
	Q_OBJECT
public:
	FixedLayerFromDatasetRepOnRandom(FixedLayerFromDataset * fixedLayer, AbstractInnerView *parent = 0);
	virtual ~FixedLayerFromDatasetRepOnRandom();

	FixedLayerFromDataset* fixedLayer() const;

	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	IData* data() const override;
	void setLayerColor(QColor);
	virtual TypeRep getTypeGraphicRep() override;
    virtual void deleteLayer() override;
private:
	FixedLayerFromDatasetPropPanelOnRandom* m_propPanel;
	FixedLayerFromDatasetLayerOnRandom* m_layer;
	FixedLayerFromDataset* m_fixedLayer;
};

#endif

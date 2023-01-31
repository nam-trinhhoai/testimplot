#ifndef FixedRGBLayersFromDatasetAndCubeRepOnRandom_H
#define FixedRGBLayersFromDatasetAndCubeRepOnRandom_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"

class FixedRGBLayersFromDatasetAndCubePropPanelOnRandom;
class FixedRGBLayersFromDatasetAndCubeLayerOnRandom;
class FixedRGBLayersFromDatasetAndCube;
class IGeorefImage;

class FixedRGBLayersFromDatasetAndCubeRepOnRandom : public AbstractGraphicRep,
	public ISampleDependantRep {
Q_OBJECT
public:
	FixedRGBLayersFromDatasetAndCubeRepOnRandom(FixedRGBLayersFromDatasetAndCube *fixedLayer,
			AbstractInnerView *parent = 0);
	virtual ~FixedRGBLayersFromDatasetAndCubeRepOnRandom();

	FixedRGBLayersFromDatasetAndCube* fixedRGBLayersFromDataset() const;

	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;

	IData* data() const override;
	virtual TypeRep getTypeGraphicRep() override;
	virtual void deleteLayer() override;

public slots:
	void trt_changeColor();
	void trt_properties();
	void trt_location();
protected:
	FixedRGBLayersFromDatasetAndCubePropPanelOnRandom *m_propPanel;
	FixedRGBLayersFromDatasetAndCubeLayerOnRandom *m_layer;
	FixedRGBLayersFromDatasetAndCube *m_fixedLayer;
};

#endif

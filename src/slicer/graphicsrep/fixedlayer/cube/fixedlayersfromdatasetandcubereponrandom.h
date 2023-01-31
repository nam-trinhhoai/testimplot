#ifndef FixedLayersFromDatasetAndCubeRepOnRandom_H
#define FixedLayersFromDatasetAndCubeRepOnRandom_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"

class FixedLayersFromDatasetAndCubePropPanelOnRandom;
class FixedLayersFromDatasetAndCubeLayerOnRandom;
class FixedLayersFromDatasetAndCube;
class IGeorefImage;

class FixedLayersFromDatasetAndCubeRepOnRandom : public AbstractGraphicRep,
	public ISampleDependantRep {
Q_OBJECT
public:
	FixedLayersFromDatasetAndCubeRepOnRandom(FixedLayersFromDatasetAndCube *fixedLayer,
			AbstractInnerView *parent = 0);
	virtual ~FixedLayersFromDatasetAndCubeRepOnRandom();

	FixedLayersFromDatasetAndCube* fixedLayersFromDataset() const;

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
	void trt_location();
	void trt_changeColor();
	void trt_properties();

protected:
	FixedLayersFromDatasetAndCubePropPanelOnRandom *m_propPanel;
	FixedLayersFromDatasetAndCubeLayerOnRandom *m_layer;
	FixedLayersFromDatasetAndCube *m_fixedLayer;
};

#endif

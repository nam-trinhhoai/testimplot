#ifndef FixedLayersFromDatasetAndCubeRepOnSlice_H
#define FixedLayersFromDatasetAndCubeRepOnSlice_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"


class FixedLayersFromDatasetAndCubePropPanelOnSlice;
class FixedLayersFromDatasetAndCubeLayerOnSlice;
class FixedLayersFromDatasetAndCube;
class IGeorefImage;

class FixedLayersFromDatasetAndCubeRepOnSlice : public AbstractGraphicRep,  public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
	FixedLayersFromDatasetAndCubeRepOnSlice(FixedLayersFromDatasetAndCube *fixedLayer, SliceDirection dir, AbstractInnerView *parent = 0);
	virtual ~FixedLayersFromDatasetAndCubeRepOnSlice();

	FixedLayersFromDatasetAndCube* fixedLayersFromDataset() const;

	void setSliceIJPosition(int imageVal) override;
	int currentIJSliceRep()const{return m_currentSlice;}
	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;

	IData* data() const override;
	SliceDirection direction() const {return m_dir;}
	virtual TypeRep getTypeGraphicRep() override;

public slots:
	void trt_location();
	void trt_changeColor();
	void trt_properties();


public:
	FixedLayersFromDatasetAndCubePropPanelOnSlice *m_propPanel;
	FixedLayersFromDatasetAndCubeLayerOnSlice *m_layer;
	FixedLayersFromDatasetAndCube *m_fixedLayer;


	SliceDirection m_dir;
	int m_currentSlice;
};

#endif

#ifndef FixedRGBLayersFromDatasetAndCubeRepOnSlice_H
#define FixedRGBLayersFromDatasetAndCubeRepOnSlice_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"

class FixedRGBLayersFromDatasetAndCubePropPanelOnSlice;
class FixedRGBLayersFromDatasetAndCubeLayerOnSlice;
class FixedRGBLayersFromDatasetAndCube;
class IGeorefImage;
class QMenu;

class FixedRGBLayersFromDatasetAndCubeRepOnSlice : public AbstractGraphicRep,  public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
	FixedRGBLayersFromDatasetAndCubeRepOnSlice(FixedRGBLayersFromDatasetAndCube *fixedLayer, SliceDirection dir, AbstractInnerView *parent = 0);
	virtual ~FixedRGBLayersFromDatasetAndCubeRepOnSlice();

	FixedRGBLayersFromDatasetAndCube* fixedRGBLayersFromDataset() const;

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
	void buildContextMenu(QMenu* menu);


protected:
	FixedRGBLayersFromDatasetAndCubePropPanelOnSlice *m_propPanel;
	FixedRGBLayersFromDatasetAndCubeLayerOnSlice *m_layer;
	FixedRGBLayersFromDatasetAndCube *m_fixedLayer;

	SliceDirection m_dir;
	int m_currentSlice;

	public slots:
	void trt_changeColor();
	void trt_properties();
	void trt_location();
	void computeGccOnSpectrum();
};

#endif

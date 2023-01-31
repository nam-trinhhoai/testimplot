#ifndef FixedRGBLayersFromDatasetRepOnSlice_H
#define FixedRGBLayersFromDatasetRepOnSlice_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"

class FixedRGBLayersFromDatasetPropPanelOnSlice;
class FixedRGBLayersFromDatasetLayerOnSlice;
class FixedRGBLayersFromDataset;
class IGeorefImage;

class FixedRGBLayersFromDatasetRepOnSlice : public AbstractGraphicRep,  public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
	FixedRGBLayersFromDatasetRepOnSlice(FixedRGBLayersFromDataset *fixedLayer, const IGeorefImage * const transfoProvider, SliceDirection dir, AbstractInnerView *parent = 0);
	virtual ~FixedRGBLayersFromDatasetRepOnSlice();

	FixedRGBLayersFromDataset* fixedRGBLayersFromDataset() const;

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

protected:
	FixedRGBLayersFromDatasetPropPanelOnSlice *m_propPanel;
	FixedRGBLayersFromDatasetLayerOnSlice *m_layer;
	FixedRGBLayersFromDataset *m_fixedLayer;

	SliceDirection m_dir;
	int m_currentSlice;

	const IGeorefImage * const m_transfoProvider;
};

#endif

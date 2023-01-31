#ifndef FixedLayerFromDatasetRepOnSlice_H
#define FixedLayerFromDatasetRepOnSlice_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"

#include <QColor>

class FixedLayerFromDatasetPropPanelOnSlice;
class FixedLayerFromDatasetLayerOnSlice;
class FixedLayerFromDataset;
class IGeorefImage;

//For Section
class FixedLayerFromDatasetRepOnSlice: public AbstractGraphicRep,  public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
FixedLayerFromDatasetRepOnSlice(FixedLayerFromDataset * fixedLayer, const IGeorefImage * const transfoProvider, SliceDirection dir, AbstractInnerView *parent = 0);
	virtual ~FixedLayerFromDatasetRepOnSlice();

	FixedLayerFromDataset* fixedLayer() const;

	void setSliceIJPosition(int imageVal) override;
	int currentIJSliceRep()const{return m_currentSlice;}
	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	IData* data() const override;
	void setLayerColor(QColor);
	virtual TypeRep getTypeGraphicRep() override;
private:
	FixedLayerFromDatasetPropPanelOnSlice *m_propPanel;
	FixedLayerFromDatasetLayerOnSlice *m_layer;
	FixedLayerFromDataset * m_fixedLayer;

	SliceDirection m_dir;
	int m_currentSlice;
	const IGeorefImage * const m_transfoProvider;
};

#endif

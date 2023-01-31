#ifndef StratiSliceAttributeRepOnSlice_H
#define StratiSliceAttributeRepOnSlice_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"

class StratiSliceAttributePropPanelOnSlice;
class StratiSliceAttributeSliceLayer;
class AbstractStratiSliceAttribute;
class IGeorefImage;

class StratiSliceAttributeRepOnSlice: public AbstractGraphicRep,  public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
	StratiSliceAttributeRepOnSlice(AbstractStratiSliceAttribute *stratislice, const IGeorefImage * const transfoProvider, SliceDirection dir, AbstractInnerView *parent = 0);
	virtual ~StratiSliceAttributeRepOnSlice();

	AbstractStratiSliceAttribute* stratiSliceAttribute() const;

	void setSliceIJPosition(int imageVal) override;
	int currentIJSliceRep()const{return m_currentSlice;}
	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual TypeRep getTypeGraphicRep() override;
	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;

	IData* data() const override;

protected:
	StratiSliceAttributePropPanelOnSlice *m_propPanel;
	StratiSliceAttributeSliceLayer *m_layer;
	AbstractStratiSliceAttribute *m_stratislice;

	SliceDirection m_dir;
	int m_currentSlice;

	const IGeorefImage * const m_transfoProvider;
};

#endif

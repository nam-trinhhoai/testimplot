#ifndef LayerRGTRepOnSlice_H
#define LayerRGTRepOnSlice_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"

class LayerRGTPropPanelOnSlice;
class LayerRGTSliceLayer;
class LayerSlice;
class IGeorefImage;

//For Section
class LayerRGTRepOnSlice: public AbstractGraphicRep,  public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
	LayerRGTRepOnSlice(LayerSlice * layerslice, const IGeorefImage * const transfoProvider, SliceDirection dir, AbstractInnerView *parent = 0);
	virtual ~LayerRGTRepOnSlice();

	LayerSlice* layerSlice() const;

	void setSliceIJPosition(int imageVal) override;
	int currentIJSliceRep()const{return m_currentSlice;}
	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const;
	virtual void buildContextMenu(QMenu *menu) override;
	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	IData* data() const override;
	virtual TypeRep getTypeGraphicRep() override;
signals:
		void deletedRep(AbstractGraphicRep *rep);// MZR 15072021
private slots:
	void deleteLayerRGTRepOnSlice();
private:
	LayerRGTPropPanelOnSlice *m_propPanel;
	LayerRGTSliceLayer *m_layer;
	LayerSlice * m_layerslice;

	SliceDirection m_dir;
	int m_currentSlice;
	const IGeorefImage * const m_transfoProvider;
};

#endif

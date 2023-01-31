#ifndef LayerRGTRep_H
#define LayerRGTRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"

class CUDAImagePaletteHolder;
class QGLLineItem;
class LayerRGTPropPanel;
class LayerRGTLayer;
class LayerRGT3DLayer;
class LayerSlice;

//For BaseMap
class LayerRGTRep: public AbstractGraphicRep, public IMouseImageDataProvider, public ISampleDependantRep {
Q_OBJECT
public:
	LayerRGTRep(LayerSlice * layerSlice, AbstractInnerView *parent = 0);
	virtual ~LayerRGTRep();

	LayerSlice* layerSlice() const;

	void showCrossHair(bool val);
	bool crossHair() const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;

	IData* data() const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y,MouseInfo & info) override;

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual void buildContextMenu(QMenu *menu) override;
	virtual TypeRep getTypeGraphicRep() override;
	signals:
	void deletedRep(AbstractGraphicRep *rep);// MZR 15072021

private slots:
	void dataChanged();
	void deleteLayerRGTRep();
private:
	LayerRGTPropPanel *m_propPanel;
	LayerRGTLayer *m_layer;
	LayerRGT3DLayer *m_layer3D;

	LayerSlice * m_layerslice;

	bool m_showCrossHair;
};

#endif

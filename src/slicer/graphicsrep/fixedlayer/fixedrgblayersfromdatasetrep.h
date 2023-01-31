#ifndef FixedRGBLayersFromDatasetRep_H
#define FixedRGBLayersFromDatasetRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"

class CUDAImagePaletteHolder;
class CUDARGBImage;
class QGraphicsObject;

class FixedRGBLayersFromDatasetPropPanel;
class FixedRGBLayersFromDataset3DLayer;
class FixedRGBLayersFromDatasetLayerOnMap;
class FixedRGBLayersFromDataset;

class QGLLineItem;

//Representation d'une slice RGB
class FixedRGBLayersFromDatasetRep: public AbstractGraphicRep,
		public IMouseImageDataProvider, public ISampleDependantRep {
Q_OBJECT
public:
	FixedRGBLayersFromDatasetRep(FixedRGBLayersFromDataset *layer, AbstractInnerView *parent = 0);
	virtual ~FixedRGBLayersFromDatasetRep();

	FixedRGBLayersFromDataset* fixedRGBLayersFromDataset() const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;
	Graphic3DLayer* layer3D(QWindow *parent, Qt3DCore::QEntity *root,
			Qt3DRender::QCamera *camera) override;

	IData* data() const override;

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y, MouseInfo &info) override;
	virtual TypeRep getTypeGraphicRep() override;
private slots:
	void dataChangedRed();
	void dataChangedGreen();
	void dataChangedBlue();
protected:
	FixedRGBLayersFromDatasetPropPanel *m_propPanel;
	FixedRGBLayersFromDataset3DLayer *m_layer3D;
	FixedRGBLayersFromDatasetLayerOnMap *m_layer;

	FixedRGBLayersFromDataset *m_data;
};

#endif

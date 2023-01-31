#ifndef FixedRGBLayersFromDatasetAndCubeRep_H
#define FixedRGBLayersFromDatasetAndCubeRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"
#include "iGraphicToolDataControl.h"
#include "iCUDAImageClone.h"

class CUDAImagePaletteHolder;
class CUDARGBImage;
class QGraphicsObject;

class FixedRGBLayersFromDatasetAndCubePropPanel;
class FixedRGBLayersFromDatasetAndCube3DLayer;
class FixedRGBLayersFromDatasetAndCubeLayerOnMap;
class FixedRGBLayersFromDatasetAndCube;

class QGLLineItem;

//Representation d'une slice RGB
class FixedRGBLayersFromDatasetAndCubeRep: public AbstractGraphicRep,
		public IMouseImageDataProvider, public ISampleDependantRep,
		public iGraphicToolDataControl, public iCUDAImageClone {
Q_OBJECT
public:
	FixedRGBLayersFromDatasetAndCubeRep(FixedRGBLayersFromDatasetAndCube *layer, AbstractInnerView *parent = 0);
	virtual ~FixedRGBLayersFromDatasetAndCubeRep();

	FixedRGBLayersFromDatasetAndCube* fixedRGBLayersFromDataset() const;

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

	// iGraphicToolDataControl
	void deleteGraphicItemDataContent(QGraphicsItem *item) override;

	QGraphicsObject* cloneCUDAImageWithMask(QGraphicsItem *parent);

	void buildContextMenu(QMenu* menu);


private slots:
	void dataChangedAll();
	void dataChangedRed();
	void dataChangedGreen();
	void dataChangedBlue();
	void computeGccOnSpectrum();

protected:
	FixedRGBLayersFromDatasetAndCubePropPanel *m_propPanel;
	FixedRGBLayersFromDatasetAndCube3DLayer *m_layer3D;
	FixedRGBLayersFromDatasetAndCubeLayerOnMap *m_layer;

	FixedRGBLayersFromDatasetAndCube *m_data;
};

#endif

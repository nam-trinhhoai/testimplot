#ifndef FixedLayersFromDatasetAndCubeRep_H
#define FixedLayersFromDatasetAndCubeRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"

class CUDAImagePaletteHolder;
class QGraphicsObject;

class FixedLayersFromDatasetAndCubePropPanel;
class FixedLayersFromDatasetAndCube3DLayer;
class FixedLayersFromDatasetAndCubeLayerOnMap;
class FixedLayersFromDatasetAndCube;

class QGLLineItem;

//Representation d'une slice
class FixedLayersFromDatasetAndCubeRep: public AbstractGraphicRep,
		public IMouseImageDataProvider, public ISampleDependantRep {
Q_OBJECT
public:
	FixedLayersFromDatasetAndCubeRep(FixedLayersFromDatasetAndCube *layer, AbstractInnerView *parent = 0);
	virtual ~FixedLayersFromDatasetAndCubeRep();

	FixedLayersFromDatasetAndCube* fixedLayersFromDataset() const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;
	Graphic3DLayer* layer3D(QWindow *parent, Qt3DCore::QEntity *root,
			Qt3DRender::QCamera *camera) override;

	IData* data() const override;
	virtual TypeRep getTypeGraphicRep() override;

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y, MouseInfo &info) override;
private slots:
	void dataChanged();
protected:
	FixedLayersFromDatasetAndCubePropPanel *m_propPanel = nullptr;
	FixedLayersFromDatasetAndCube3DLayer *m_layer3D = nullptr;
	FixedLayersFromDatasetAndCubeLayerOnMap *m_layer = nullptr;

	FixedLayersFromDatasetAndCube *m_data = nullptr;
};

#endif

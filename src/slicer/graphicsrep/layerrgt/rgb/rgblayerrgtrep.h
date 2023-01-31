#ifndef RGBLayerRGTRep_H
#define RGBLayerRGTRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"
#include "iGraphicToolDataControl.h"
#include "iCUDAImageCloneBaseMap.h"

class CUDAImagePaletteHolder;
class CUDARGBImage;
class QGraphicsObject;

class RGBLayerRGTPropPanel;
class RGBLayerRGTLayer;
class RGBLayerRGT3DLayer;
class ImagePositionControler;
class RGBLayerSlice;

class QGLLineItem;
class BaseMapSurface;

//Representation d'une slice RGB
class RGBLayerRGTRep: public AbstractGraphicRep,
		public IMouseImageDataProvider,
		public ISampleDependantRep,
		public iGraphicToolDataControl,
		public iCUDAImageCloneBaseMap {
Q_OBJECT
public:
	RGBLayerRGTRep(RGBLayerSlice *stratislice, AbstractInnerView *parent = 0);
	virtual ~RGBLayerRGTRep();

	RGBLayerSlice* rgbLayerSlice() const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;
	Graphic3DLayer* layer3D(QWindow *parent, Qt3DCore::QEntity *root,
			Qt3DRender::QCamera *camera) override;

	IData* data() const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y, MouseInfo &info) override;

	// iGraphicToolDataControl
	void deleteGraphicItemDataContent(QGraphicsItem *item) override;

	// iCUDAImageClone
	QGraphicsObject* cloneCUDAImageWithMask(QGraphicsItem *parent) override;

	// iCUDAImageCloneBaseMap
	virtual BaseMapSurface* cloneCUDAImageWithMaskOnBaseMap(QGraphicsItem *parent) override;

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual void buildContextMenu(QMenu * menu) override;
	virtual TypeRep getTypeGraphicRep() override;
private slots:
	void dataChangedRed();
	void dataChangedGreen();
	void dataChangedBlue();
	void deleteRGBLayerRGTRep();
signals:
	void deletedRep(AbstractGraphicRep *rep);// MZR 17082021
protected:
	RGBLayerRGTPropPanel *m_propPanel;
	RGBLayerRGTLayer *m_layer;
	RGBLayerRGT3DLayer *m_layer3D;

	RGBLayerSlice *m_rgbLayerSlice;
};

#endif

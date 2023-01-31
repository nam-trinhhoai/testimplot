#ifndef StratiSliceRGBAttributeRep_H
#define StratiSliceRGBAttributeRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"

class CUDAImagePaletteHolder;
class CUDARGBImage;
class QGraphicsObject;

class StratiSliceRGBPropPanel;
class StratiSliceRGBLayer;
class StratiSliceRGB3DLayer;
class ImagePositionControler;
class QGLLineItem;
class RGBStratiSliceAttribute;

//Representation d'une slice RGB
class StratiSliceRGBAttributeRep: public AbstractGraphicRep,
		public IMouseImageDataProvider, public ISampleDependantRep {
Q_OBJECT
public:
	StratiSliceRGBAttributeRep(RGBStratiSliceAttribute *stratislice, AbstractInnerView *parent = 0);
	virtual ~StratiSliceRGBAttributeRep();

	RGBStratiSliceAttribute* stratiSliceAttribute() const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;
	Graphic3DLayer* layer3D(QWindow *parent, Qt3DCore::QEntity *root,
			Qt3DRender::QCamera *camera) override;

	IData* data() const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y, MouseInfo &info) override;

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual TypeRep getTypeGraphicRep() override;
private slots:
	void dataChangedRed();
	void dataChangedGreen();
	void dataChangedBlue();
protected:
	StratiSliceRGBPropPanel *m_propPanel;
	StratiSliceRGBLayer *m_layer;
	StratiSliceRGB3DLayer *m_layer3D;

	RGBStratiSliceAttribute *m_stratislice;
};

#endif

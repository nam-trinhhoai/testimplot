#ifndef StratiSliceAmplitudeRep_H
#define StratiSliceAmplitudeRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"

class CUDAImagePaletteHolder;
class QGLLineItem;
class StratiSliceAmplitudePropPanel;
class StratiSliceAmplitudeLayer;
class StratiSliceAmplitude3DLayer;
class AmplitudeStratiSliceAttribute;

class StratiSliceAmplitudeRep: public AbstractGraphicRep, public IMouseImageDataProvider,
		public ISampleDependantRep {
Q_OBJECT
public:
	StratiSliceAmplitudeRep(AmplitudeStratiSliceAttribute * stratislice, AbstractInnerView *parent = 0);
	virtual ~StratiSliceAmplitudeRep();

	AmplitudeStratiSliceAttribute* stratiSliceAttribute() const;

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
	virtual TypeRep getTypeGraphicRep() override;
private slots:
	void dataChanged();
private:
	StratiSliceAmplitudePropPanel *m_propPanel;
	StratiSliceAmplitudeLayer *m_layer;
	StratiSliceAmplitude3DLayer *m_layer3D;

	AmplitudeStratiSliceAttribute * m_stratislice;

	bool m_showCrossHair;
};

#endif

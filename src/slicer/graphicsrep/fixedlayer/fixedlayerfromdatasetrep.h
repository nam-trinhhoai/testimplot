#ifndef FixedLayerFromDatasetRep_H
#define FixedLayerFromDatasetRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"
#include "iGraphicToolDataControl.h"

class CUDAImagePaletteHolder;
class QGLLineItem;
class FixedLayerFromDatasetPropPanel;
class FixedLayerFromDatasetLayer;
class FixedLayerFromDataset;

//For BaseMap
class FixedLayerFromDatasetRep: public AbstractGraphicRep, public IMouseImageDataProvider, public ISampleDependantRep,
		public iGraphicToolDataControl {
Q_OBJECT
public:
	FixedLayerFromDatasetRep(FixedLayerFromDataset * layerSlice, AbstractInnerView *parent = 0);
	virtual ~FixedLayerFromDatasetRep();

	FixedLayerFromDataset* fixedLayer() const;

	void showCrossHair(bool val);
	bool crossHair() const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	IData* data() const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y,MouseInfo & info) override;

	void chooseAttribute(QString attributeName);

	CUDAImagePaletteHolder* image() {
		return m_currentAttribute;
	}

	CUDAImagePaletteHolder* isoSurfaceHolder() {
		return m_currentIso;
	}

	QString isoName() const {
		return m_currentIsoName;
	}
	QString attributeName() const {
		return m_currentAttributeName;
	}

	virtual QString name() const override;

	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual TypeRep getTypeGraphicRep() override;

	virtual void deleteGraphicItemDataContent(QGraphicsItem* item) override;
private slots:
	void dataChanged();
	void updateAttribute(QString propName);
	void updateIso(QString propName);
	void initProperties(QString propName);
private:
	void chooseIsochrone();
	void updateAttributePalette();

	FixedLayerFromDatasetPropPanel *m_propPanel = nullptr;
	FixedLayerFromDatasetLayer *m_layer = nullptr;

	FixedLayerFromDataset * m_fixedLayer;

	bool m_showCrossHair = false;
	QString m_currentIsoName;
	CUDAImagePaletteHolder* m_currentIso = nullptr;
	QString m_currentAttributeName;
	CUDAImagePaletteHolder* m_currentAttribute = nullptr;
};

#endif

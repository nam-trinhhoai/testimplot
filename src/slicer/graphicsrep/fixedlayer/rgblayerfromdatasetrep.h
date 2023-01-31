#ifndef RgbLayerFromDatasetRep_H
#define RgbLayerFromDatasetRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"

class CUDAImagePaletteHolder;
class QGLLineItem;
class RgbLayerFromDatasetPropPanel;
class RgbLayerFromDatasetLayer;
class RgbLayerFromDataset;
class CUDARGBImage;

//For BaseMap
class RgbLayerFromDatasetRep: public AbstractGraphicRep, public IMouseImageDataProvider, public ISampleDependantRep {
Q_OBJECT
public:
	RgbLayerFromDatasetRep(RgbLayerFromDataset * layerSlice, AbstractInnerView *parent = 0);
	virtual ~RgbLayerFromDatasetRep();

	RgbLayerFromDataset* fixedLayer() const;

	void showCrossHair(bool val);
	bool crossHair() const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	IData* data() const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y,MouseInfo & info) override;

	void chooseRed(QString attributeName);
	void chooseGreen(QString attributeName);
	void chooseBlue(QString attributeName);

	CUDARGBImage* image() {
		return m_currentAttribute;
	}

	CUDAImagePaletteHolder* isoSurfaceHolder() {
		return m_currentIso;
	}

	QString isoName() const {
		return m_currentIsoName;
	}
	QString redName() const {
		return m_currentRedName;
	}
	QString greenName() const {
		return m_currentGreenName;
	}
	QString blueName() const {
		return m_currentBlueName;
	}

	virtual QString name() const override;

	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual TypeRep getTypeGraphicRep() override;

private slots:
	void dataChanged();
	void updateRed(QString propName);
	void updateGreen(QString propName);
	void updateBlue(QString propName);
	void updateIso(QString propName);
	void initProperties(QString propName);
private:
	void chooseIsochrone();
	void updateRedPalette();
	void updateGreenPalette();
	void updateBluePalette();

	RgbLayerFromDatasetPropPanel *m_propPanel = nullptr;
	RgbLayerFromDatasetLayer *m_layer = nullptr;

	RgbLayerFromDataset * m_fixedLayer;

	bool m_showCrossHair;
	QString m_currentIsoName;
	CUDAImagePaletteHolder* m_currentIso = nullptr;
	QString m_currentRedName;
	QString m_currentGreenName;
	QString m_currentBlueName;
	CUDARGBImage* m_currentAttribute = nullptr;
};

#endif

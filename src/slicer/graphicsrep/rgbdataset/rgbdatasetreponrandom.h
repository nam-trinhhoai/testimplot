#ifndef RgbDatasetRepOnRandom_H
#define RgbDatasetRepOnRandom_H

#include <QObject>
#include <QVector2D>
#include <QUuid>
#include <QMap>
#include <QPair>
#include <QMutex>
#include <QPolygon>
#include <memory>
#include "abstractgraphicrep.h"
#include "sliceutils.h"
#include "imouseimagedataprovider.h"
#include "affinetransformation.h"
#include "isampledependantrep.h"
#include "isliceablerep.h"

class CUDAImagePaletteHolder;
class RgbDatasetPropPanelOnRandom;
class RgbDatasetLayerOnRandom;
class RgbDataset;
class Affine2DTransformation;
class SpectralImageCache;

class RgbDatasetRepOnRandom:
		public AbstractGraphicRep,
		public IMouseImageDataProvider,
		public ISampleDependantRep {
Q_OBJECT
public:
	RgbDatasetRepOnRandom(RgbDataset *data, AbstractInnerView *parent = 0);
	virtual ~RgbDatasetRepOnRandom();

	CUDAImagePaletteHolder* red() {
		return m_red;
	}

	CUDAImagePaletteHolder* green() {
		return m_green;
	}

	CUDAImagePaletteHolder* blue() {
		return m_blue;
	}

	CUDAImagePaletteHolder* alpha() {
		return m_alpha;
	}

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;
	IData* data() const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y, MouseInfo &info) override;

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	// used if alpha == nullptr
	float getOpacity() const;
	void setOpacity(float);

	RgbDataset* rgbDataset() const;
	virtual TypeRep getTypeGraphicRep() override;
signals:
	void channelChanged(int channel);
	void opacityChanged(); // used if alpha == nullptr
private slots:
	void dataChanged();
	void redChannelChanged();
	void greenChannelChanged();
	void blueChannelChanged();
	void alphaChannelChanged();
private:
	void refreshLayer();
	void loadRandom();
	void createCache();
private:
	bool createImagePaletteHolder();

	CUDAImagePaletteHolder* m_red;
	CUDAImagePaletteHolder* m_green;
	CUDAImagePaletteHolder* m_blue;
	CUDAImagePaletteHolder* m_alpha;
	Affine2DTransformation* m_transformation;

	RgbDatasetPropPanelOnRandom *m_propPanel;

	RgbDatasetLayerOnRandom *m_layer;

	RgbDataset *m_data;
	float m_opacity;// used if alpha == nullptr
	QPolygon m_discreatePolygon;

	std::shared_ptr<SpectralImageCache> m_redCache;
	std::shared_ptr<SpectralImageCache> m_greenCache;
	std::shared_ptr<SpectralImageCache> m_blueCache;
	std::shared_ptr<SpectralImageCache> m_alphaCache;
};

#endif

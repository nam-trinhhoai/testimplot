#ifndef RgbDatasetRepOnSlice_H
#define RgbDatasetRepOnSlice_H

#include <QObject>
#include <QVector2D>
#include <QUuid>
#include <QMap>
#include <QPair>
#include <QMutex>
#include <memory>
#include "abstractgraphicrep.h"
#include "sliceutils.h"
#include "imouseimagedataprovider.h"
#include "affinetransformation.h"
#include "isampledependantrep.h"
#include "isliceablerep.h"

class CUDAImagePaletteHolder;
class RgbDatasetPropPanelOnSlice;
class RgbDatasetLayerOnSlice;
class RgbDataset;
class SpectralImageCache;

class RgbDatasetRepOnSlice:
		public AbstractGraphicRep,
		public IMouseImageDataProvider,
		public ISampleDependantRep,
		public ISliceableRep {
Q_OBJECT
public:
	RgbDatasetRepOnSlice(RgbDataset *data, CUDAImagePaletteHolder* red, CUDAImagePaletteHolder* green,
			CUDAImagePaletteHolder* blue, CUDAImagePaletteHolder* alphaImage,
			const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo, SliceDirection dir =
					SliceDirection::Inline, AbstractInnerView *parent = 0);
	virtual ~RgbDatasetRepOnSlice();

	int currentSliceWorldPosition() const;
	int currentSliceIJPosition() const;

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

	SliceDirection direction() const {
		return m_dir;
	}

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;
	IData* data() const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y, MouseInfo &info) override;

	QPair<QVector2D,AffineTransformation> sliceRangeAndTransfo() const;


	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual TypeRep getTypeGraphicRep() override;
	// used if alpha == nullptr
	float getOpacity() const;
	void setOpacity(float);

	RgbDataset* rgbDataset() const;

public slots:
	void setSliceWorldPosition(int pos, bool force=false);
	void setSliceIJPosition(int pos, bool force);
	virtual void setSliceIJPosition(int val) override;
	void redChannelChanged();
	void greenChannelChanged();
	void blueChannelChanged();
	void alphaChannelChanged();
signals:
	void sliceWordPositionChanged(int pos);
	void sliceIJPositionChanged(int pos);
	void channelChanged(int channel);
	void opacityChanged(); // used if alpha == nullptr
private slots:
	void dataChanged();
private:
	void refreshLayer();
	void loadSlice(unsigned int z);
	void createCache();
private:
	unsigned int m_currentSlice;
	bool m_isCurrentSliceLoaded;
	CUDAImagePaletteHolder* m_red;
	CUDAImagePaletteHolder* m_green;
	CUDAImagePaletteHolder* m_blue;
	CUDAImagePaletteHolder* m_alpha;
	SliceDirection m_dir;

	RgbDatasetPropPanelOnSlice *m_propPanel;

	RgbDatasetLayerOnSlice *m_layer;

	//Range min/max of the data available
	QPair<QVector2D,AffineTransformation>  m_sliceRangeAndTransfo;

	RgbDataset *m_data;
	float m_opacity;// used if alpha == nullptr

	std::shared_ptr<SpectralImageCache> m_redCache;
	std::shared_ptr<SpectralImageCache> m_greenCache;
	std::shared_ptr<SpectralImageCache> m_blueCache;
	std::shared_ptr<SpectralImageCache> m_alphaCache;
};

#endif

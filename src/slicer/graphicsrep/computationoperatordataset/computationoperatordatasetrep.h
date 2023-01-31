#ifndef ComputationOperatorDatasetRep_H
#define ComputationOperatorDatasetRep_H

#include <QObject>
#include <QVector2D>
#include <QUuid>
#include <QMap>
#include <QPair>
#include <QMutex>
#include <memory>
#include "abstractgraphicrep.h"
#include "idatacontrolerholder.h"
#include "sliceutils.h"
#include "idatacontrolerprovider.h"
#include "imouseimagedataprovider.h"
#include "affinetransformation.h"
#include "isampledependantrep.h"
#include "iGraphicToolDataControl.h"
#include "iCUDAImageClone.h"
#include "isliceablerep.h"

class CUDAImagePaletteHolder;
class QGLLineItem;
class ComputationOperatorDatasetPropPanel;
class ComputationOperatorDatasetLayer;
class DataControler;
class ComputationOperatorDataset;
class ArraySpectralImageCache;

class ComputationOperatorDatasetRep:
		public AbstractGraphicRep,
		public IMouseImageDataProvider,
		public ISampleDependantRep,
		public iGraphicToolDataControl,
		public iCUDAImageClone,
		public ISliceableRep {
Q_OBJECT
public:
	ComputationOperatorDatasetRep(ComputationOperatorDataset *data, CUDAImagePaletteHolder *slice,
			const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo, SliceDirection dir =
					SliceDirection::Inline, AbstractInnerView *parent = 0);
	virtual ~ComputationOperatorDatasetRep();

	int currentSliceWorldPosition() const;
	int currentSliceIJPosition() const;


	CUDAImagePaletteHolder* image() {
		return m_image;
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

	// iGraphicToolDataControl
	void deleteGraphicItemDataContent(QGraphicsItem *item) override;

	QGraphicsObject* cloneCUDAImageWithMask(QGraphicsItem *parent) override;

	QPair<QVector2D,AffineTransformation> sliceRangeAndTransfo() const;

	void showColorScale(bool val);
	bool colorScale() const;
	int channel() const;

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual TypeRep getTypeGraphicRep() override;
	const std::vector<std::vector<char>>& lockCache();
	void unlockCache();

	virtual void setSliceIJPosition(int val) override;

public slots:
	void setSliceWorldPosition(int pos, bool force=false);
	void setSliceIJPosition(int pos, bool force);
	void setChannel(int channel);
signals:
	void sliceWordPositionChanged(int pos);
	void sliceIJPositionChanged(int pos);
	void channelChanged(int channel);
	void layerHidden();

private slots:
	void dataChanged();
	void rangeLockChanged();
	void relayHidden();
private:
	void refreshLayer();
	void loadSlice(unsigned int z);
private:
	unsigned int m_currentSlice = 0;
	CUDAImagePaletteHolder *m_image = nullptr;
	SliceDirection m_dir = SliceDirection::Inline;

	ComputationOperatorDatasetPropPanel *m_propPanel = nullptr;

	ComputationOperatorDatasetLayer *m_layer = nullptr;

	//Range min/max of the data available
	QPair<QVector2D,AffineTransformation>  m_sliceRangeAndTransfo;

	ComputationOperatorDataset *m_data = nullptr;
	int m_channel = 0;

	QMap<DataControler*, QGraphicsItem*> m_datacontrolers;

	bool m_showColorScale = false;
	std::unique_ptr<ArraySpectralImageCache> m_cache;
	QMutex m_cacheMutex;
};

#endif

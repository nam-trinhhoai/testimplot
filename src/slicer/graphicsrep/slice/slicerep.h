#ifndef SliceRep_H
#define SliceRep_H

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

class CUDAImagePaletteHolder;
class QGLLineItem;
class SlicePropPanel;
class SliceLayer;
class DataControler;
class Seismic3DAbstractDataset;
class MonoBlockSpectralImageCache;

class SliceRep:
		public AbstractGraphicRep,
		public IDataControlerHolder,
		public IDataControlerProvider,
		public IMouseImageDataProvider,
		public ISampleDependantRep,
		public iGraphicToolDataControl,
		public iCUDAImageClone {
Q_OBJECT
public:
	SliceRep(Seismic3DAbstractDataset *data, CUDAImagePaletteHolder *slice,
			const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo, SliceDirection dir =
					SliceDirection::Inline, AbstractInnerView *parent = 0);
	virtual ~SliceRep();

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

	//IDataControlerHolder
	QGraphicsItem* getOverlayItem(DataControler *controler,
			QGraphicsItem *parent) override;
	QGraphicsItem* releaseOverlayItem(DataControler *controler) override;

	//IDataControlerProvider
	virtual void setDataControler(DataControler *controler) override;
	virtual DataControler* dataControler() const override;

	virtual void notifyDataControlerMouseMoved(double worldX, double worldY,
			Qt::MouseButton button, Qt::KeyboardModifiers keys) override;
	virtual void notifyDataControlerMousePressed(double worldX, double worldY,
			Qt::MouseButton button, Qt::KeyboardModifiers keys) override;
	virtual void notifyDataControlerMouseRelease(double worldX, double worldY,
			Qt::MouseButton button, Qt::KeyboardModifiers keys) override;
	virtual void notifyDataControlerMouseDoubleClick(double worldX, double worldY,
			Qt::MouseButton button, Qt::KeyboardModifiers keys) override;

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
	virtual void buildContextMenu(QMenu *menu) override;
	virtual TypeRep getTypeGraphicRep() override;
	const std::vector<char>& lockCache();
	void unlockCache();

public slots:
	void setSliceWorldPosition(int pos, bool force=false);
	void setSliceIJPosition(int pos, bool force=false);
	void setChannel(int channel);
signals:
	void sliceWordPositionChanged(int pos);
	void sliceIJPositionChanged(int pos);
	void channelChanged(int channel);
	void deletedRep(AbstractGraphicRep *rep);// MZR 15072021
	void layerHidden();

private slots:
	void dataChanged();
	void deleteSliceRep();
	void rangeLockChanged();
	void relayHidden();
private:
	void refreshLayer();
	void loadSlice(unsigned int z);
private:
	unsigned int m_currentSlice;
	CUDAImagePaletteHolder *m_image;
	SliceDirection m_dir;

	SlicePropPanel *m_propPanel;

	SliceLayer *m_layer;

	DataControler *m_controler;

	//Range min/max of the data available
	QPair<QVector2D,AffineTransformation>  m_sliceRangeAndTransfo;

	Seismic3DAbstractDataset *m_data;
	int m_channel;

	QMap<DataControler*, QGraphicsItem*> m_datacontrolers;

	bool m_showColorScale;
	std::unique_ptr<MonoBlockSpectralImageCache> m_cache;
	QMutex m_cacheMutex;
};

#endif

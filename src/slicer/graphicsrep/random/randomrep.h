#ifndef RandomRep_H
#define RandomRep_H

#include <QObject>
#include <QVector2D>
#include <QUuid>
#include <QMap>
#include <QPair>
#include <QPolygon>
#include <QMutex>
#include <memory>
#include "abstractgraphicrep.h"
#include "idatacontrolerholder.h"
#include "sliceutils.h"
#include "idatacontrolerprovider.h"
#include "imouseimagedataprovider.h"
#include "affinetransformation.h"
#include "lookuptable.h"
#include "isampledependantrep.h"

class CUDAImagePaletteHolder;
class QGLLineItem;
class RandomPropPanel;
class RandomLayer;
class DataControler;
class Seismic3DAbstractDataset;
class Affine2DTransformation;
class LookupTable;
class MonoBlockSpectralImageCache;

class RandomRep:
		public AbstractGraphicRep,
		public IMouseImageDataProvider,
		public ISampleDependantRep {
Q_OBJECT
public:
	RandomRep(Seismic3DAbstractDataset *data, const LookupTable& attributeLookupTable,
			AbstractInnerView *parent = 0);
	virtual ~RandomRep();

	CUDAImagePaletteHolder* image() {
		return m_image;
	}

	void cleanImage();

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;
	IData* data() const override;

	Seismic3DAbstractDataset* getdataset() const;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y, MouseInfo &info) override;

	void showColorScale(bool val);
	bool colorScale() const;
	int channel() const;
	void setChannel(int channel);

	void setPolyLine(QPolygonF);

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual void buildContextMenu(QMenu *menu) override; // MZR 19082021
	virtual TypeRep getTypeGraphicRep() override;
	virtual void deleteLayer() override;
	const std::vector<char>& lockCache();
	void unlockCache();
	//inline void setUpdatedFlag(bool bValue){m_UpdatedRep = bValue;}
	//inline bool isUpdatedFlag(){return m_UpdatedRep;}
signals:
	void channelChanged(int channel);
	void deletedRep(AbstractGraphicRep *rep);// MZR 19082021
	void layerHidden();
private slots:
	void dataChanged();
	void deleteRandomRep(); // MZR 19082021
	void rangeLockChanged();
    void relayHidden();
private:
	void refreshLayer();
	void loadRandom();
private:
	bool createImagePaletteHolder();

	CUDAImagePaletteHolder *m_image;
	LookupTable m_defaultLookupTable;
	Affine2DTransformation* m_transformation;

	RandomPropPanel *m_propPanel;

//	RandomLayer *m_layer;
	RandomLayer* m_layer;

	DataControler *m_controler;

	Seismic3DAbstractDataset *m_data;
	int m_channel;

	QMap<DataControler*, QGraphicsItem*> m_datacontrolers;

	bool m_showColorScale;
	QPolygon m_discreatePolygon;
	std::unique_ptr<MonoBlockSpectralImageCache> m_cache;
	QMutex m_cacheMutex;
//	bool m_UpdatedRep;
};

#endif

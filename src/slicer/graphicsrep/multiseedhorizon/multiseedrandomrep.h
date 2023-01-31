#ifndef MultiSeedRandomRep_H_
#define MultiSeedRandomRep_H_

#include "abstractgraphicrep.h"
#include "affinetransformation.h"
#include "sliceutils.h"
#include "RgtLayerProcessUtil.h"
#include "multiseedhorizon.h"
#include "idatacontrolerholder.h"
#include "continuouspresseventfilter.h"
#include "isampledependantrep.h"

#include <QVector2D>
#include <QPolygon>
#include <QPair>
#include <QPen>
#include <QMutex>

class MultiSeedHorizon;
class MultiSeedRandomLayer;
class GraphicLayer;
class RandomRep;
class RandomLineView;
class CUDAImagePaletteHolder;
class Seismic3DAbstractDataset;

class MultiSeedRandomRep : public AbstractGraphicRep, public IDataControlerHolder, public ISampleDependantRep
{
Q_OBJECT
public:
	enum class LOCKSTATE {
		NOLOCK, RANDOMREP, IMAGE
	};

	MultiSeedRandomRep(MultiSeedHorizon *data, AbstractInnerView *parent = 0);
	virtual ~MultiSeedRandomRep();

	static std::pair<RandomRep*, RandomRep*> findRandomRepsFromRandomInnerViewAndData(MultiSeedHorizon *data, RandomLineView *parent);

	IData* data() const override {
		return m_data;
	}

	MultiSeedHorizon* getData() const {
		return m_data;
	}

	virtual QString name() const override;

	// ISampleDependantRep
	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	// synchro from data
	void newPointCreatedSynchro(RgtSeed seed, int id);
	void pointRemovedSynchro(RgtSeed seed, int id);
	void pointMovedSynchro(RgtSeed oldSeed, RgtSeed newSeed, int id);
	void seedsResetSynchro();

	// expect coordinate with x the position in the discreatePolygon (0->polygonSize-1) and y the number of the sample (0->NumSamples-1)
	std::size_t addPoint(QPoint point); // add seed on current section
	std::size_t addPoint(int x, int y); // add seed on current section
	std::size_t addPointAndSelect(QPoint point);
	std::size_t addPointAndSelect(int x, int y);
	void moveSelectedPoint(QPointF point); // expect point in scene coordinates
	void moveSelectedPoint(QPoint point); // move point and update
	void moveSelectedPoint(int x, int y);
	void updateSeedsRepresentation();

	// graphic
	void updateGraphicRepresentation(); // ui update
	void updateMainHorizon(); // polygons update
	void updateMainHorizonNoCache();
	void updateMainHorizonNewConstrain();
	void updateDeltaHorizon();
	QPen getPen() const;
	void setPen(const QPen& pen);
	QPen getPenDelta()const;
	void setPenDelta(const QPen& pen);
	QPen getPenPoints() const;
	void setPenPoints(const QPen& pen);

	const QPolygon& getMainPolygon() const;
	const QPolygon& getTopPolygon() const;
	const QPolygon& getBottomPolygon() const;
	// JD
	const std::vector<std::vector<QPolygon>>& getReferencePolygon() const;
	// void setReferencePolygon(std::vector<QPolygon>);

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;

	//IDataControlerHolder
    virtual QGraphicsItem * getOverlayItem(DataControler * controler,QGraphicsItem *parent);
    virtual void notifyDataControlerMouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
    virtual void notifyDataControlerMousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
    virtual void notifyDataControlerMouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
    virtual void notifyDataControlerMouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);

    virtual QGraphicsItem * releaseOverlayItem(DataControler * controler);
    virtual TypeRep getTypeGraphicRep() override;
    virtual void deleteLayer() override;
private:
    template<typename InputType>
    struct CorrectSeedsFromImageKernel {
    	static void run(const void* rgtData,
    			MultiSeedRandomRep* obj, std::size_t dimI, std::size_t dimJ,
    			QList<std::tuple<RgtSeed, RgtSeed, std::size_t>>& seedsChangeList, const QPolygon& polygon);
    };

	void clearSeedsRepresentation();
	void addSeedsRepresentation();

	void clearCurveSpeedupCache();
	void moveTrackingReference(QPoint pt);
	void applyDeltaTau(int dTau);
	void applyDeltaTauRelative(double dTau, std::size_t indexLayerBottom, const std::vector<ReferenceDuo>& referenceVec,
				const std::vector<int>& referenceValues);
	void applyDTauToData();
	void initRandomReps();

	void disconnectMoveSlot(QObject* requestingObj);

	// emiting rep is supposed to take care of update itself
	void connectMoveSlotAndUpdate(QObject* requestingObj);
	void correctSeedsFromImage();

	// use rep if possible else use cacheImage
	int getValueForSeed(RandomRep* rep, std::size_t x, std::size_t y, std::size_t channel,
			CUDAImagePaletteHolder* cacheImage);
	int getValueFromSliceRepForSeed(RandomRep* rep, std::size_t x, std::size_t y, std::size_t channel);

	CUDAImagePaletteHolder* seismicImage() const;
	CUDAImagePaletteHolder* rgtImage() const;
	void updateInternalImages();
	void updateInternalImage(RandomRep* randomRep, Seismic3DAbstractDataset* dataset,
			CUDAImagePaletteHolder*& image, bool& ownCurrentImage);
	void reloadSeismicBuffer();
	void reloadRgtBuffer();

	// lock SliceRepCache or lock image buffer depending of buffers states
	// return a pointer to the slice data with seismic sample type
	// pointer is ready to use, no need to take into account the channel
	const void* lockSeismicCache() const;
	void unlockSeismicCache() const;
	const void* lockRgtCache() const;
	void unlockRgtCache() const;

	void updateSeismicAfterSwap();
	void updateRgtAfterSwap();

	void unsetSeismicRep();
	void unsetRgtRep();

	MultiSeedHorizon* m_data = nullptr;
	RandomRep* m_seismic = nullptr;
	RandomRep* m_rgt = nullptr;

	MultiSeedRandomLayer* m_layer = nullptr;

	QPolygon m_polygonMain, m_polygonTop, m_polygonBottom;
	// jd
	std::vector<std::vector<QPolygon>> m_polygonReference;
	QPolygon m_polygonMainCache;

	QPen m_penMain;
	QPen m_penDelta;
	QPen m_pointsPen;

	std::vector<RgtSeed> m_newPoints;
	std::vector<RgtSeed> m_removedPoints;
	std::vector<double> m_weightsCurve;
	std::vector<double> m_meansCurve;
	std::vector<int> m_selectedIndexCurve;
	std::vector<int> m_staticPointCurve;
	std::vector<bool> m_constrainCurve;

	ContinuousPressEventFilter m_eventFilterClass;

	//jd
	void setReferences(std::vector<float*> ref);
	std::vector<float*> m_reference;

	// locks : mandatory because of SliceRep asynchronous loadSlice
	mutable QMutex m_polygonMutex;

	BEHAVIORMODE m_oldBehaviorMode = FIXED;

	// no need to store cache because either data is valid (use cache) or image read with right channel
	bool m_ownSeismicImage = false;
	CUDAImagePaletteHolder* m_currentSeismicImage = nullptr;
	bool m_ownRgtImage = false;
	CUDAImagePaletteHolder* m_currentRgtImage = nullptr;

	// to store mode state
	mutable LOCKSTATE m_seismicLockState = LOCKSTATE::NOLOCK;
	// lock
	mutable QMutex m_seismicMutex;
	// objects used to lock on, does not take ownership
	mutable RandomRep* m_seismicSaveRep = nullptr;
	mutable CUDAImagePaletteHolder* m_seismicSaveImage = nullptr;

	// same for rgt
	mutable LOCKSTATE m_rgtLockState = LOCKSTATE::NOLOCK;
	mutable QMutex m_rgtMutex;
	mutable RandomRep* m_rgtSaveRep = nullptr;
	mutable CUDAImagePaletteHolder* m_rgtSaveImage = nullptr;
	mutable QPolygon m_cacheDiscreatePolygon;
};

#endif

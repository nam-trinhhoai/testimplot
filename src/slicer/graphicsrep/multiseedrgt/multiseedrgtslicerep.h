#ifndef MultiSeedRgtSliceRep_H_
#define MultiSeedRgtSliceRep_H_

#include "abstractgraphicrep.h"
#include "affinetransformation.h"
#include "sliceutils.h"
#include "RgtLayerProcessUtil.h"
#include "multiseedhorizon.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "idatacontrolerholder.h"
#include "continuouspresseventfilter.h"

#include <QVector2D>
#include <QPolygon>
#include <QPair>
#include <QPen>
#include <QMutex>

class MultiSeedRgt;
class MultiSeedRgtSliceLayer;
class GraphicLayer;
class SliceRep;
class AbstractSectionView;


class MultiSeedRgtSliceRep : public AbstractGraphicRep, public ISliceableRep,
	public IDataControlerHolder, public ISampleDependantRep
{
Q_OBJECT
public:
	MultiSeedRgtSliceRep(MultiSeedRgt *data,
			const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo, SliceDirection dir =
					SliceDirection::Inline, AbstractInnerView *parent = 0);
	virtual ~MultiSeedRgtSliceRep();

	// static std::pair<SliceRep*, SliceRep*> findSliceRepsFromSectionInnerViewAndData(MultiSeedRgt *data, AbstractSectionView *parent);

	// int currentSliceWorldPosition() const;
	// int currentSliceIJPosition() const;

	SliceDirection direction() const {
		return m_dir;
	}

	IData* data() const override {
		return nullptr; //m_data;
	}

	MultiSeedRgt* getData() const {
		return nullptr; // m_data;
	}

	std::size_t addPoint(QPoint point);
	std::size_t addPoint(int x, int y);


	// virtual QString name() const override;

	// virtual void setSliceIJPosition(int val) override;
	// virtual bool setSampleUnit(SampleUnit unit) override;
	// virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	// virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;


	// synchro from data
	// void newPointCreatedSynchro(RgtSeed seed, int id);
	// void pointRemovedSynchro(RgtSeed seed, int id);
	// void pointMovedSynchro(RgtSeed oldSeed, RgtSeed newSeed, int id);
	// void seedsResetSynchro();

	// std::size_t addPoint(QPoint point); // add seed on current section
	// std::size_t addPoint(int x, int y); // add seed on current section
	// std::size_t addPointAndSelect(QPoint point);
	// std::size_t addPointAndSelect(int x, int y);
	// void moveSelectedPoint(QPointF point);
	// void moveSelectedPoint(QPoint point); // move point and update
	// void moveSelectedPoint(int x, int y);
	// void updateSeedsRepresentation();

	// graphic
	// void updateGraphicRepresentation(); // ui update
	// void updateMainHorizon(); // polygons update
	// void updateMainHorizonNoCache();
	// void updateMainHorizonNewConstrain();
	// void updateDeltaHorizon();
	// QPen getPen() const;
	// void setPen(const QPen& pen);
	// QPen getPenDelta()const;
	// void setPenDelta(const QPen& pen);
	// QPen getPenPoints() const;
	// void setPenPoints(const QPen& pen);

	// const QPolygon& getMainPolygon() const;
	// const QPolygon& getTopPolygon() const;
	// const QPolygon& getBottomPolygon() const;
	// JD
	// const std::vector<std::vector<QPolygon>>& getReferencePolygon() const;
	// void setReferencePolygon(std::vector<QPolygon>);

	//AbstractGraphicRep
	// QWidget* propertyPanel() override;
	// GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) override;

	//IDataControlerHolder
    // virtual QGraphicsItem * getOverlayItem(DataControler * controler,QGraphicsItem *parent);
    // virtual void notifyDataControlerMouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
    virtual void notifyDataControlerMousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
    // virtual void notifyDataControlerMouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
    // virtual void notifyDataControlerMouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);

    // virtual QGraphicsItem * releaseOverlayItem(DataControler * controler);

    // long getMeanTauOnCurve(bool* ok);
    virtual TypeRep getTypeGraphicRep() override;
private:
    // template<typename InputType>
    // struct CorrectSeedsFromImageKernel {
    //	static void run(const void* rgtData, int channelRgt, MultiSeedRgtSliceRep* obj, std::size_t dimI,
    //			std::size_t dimJ, QList<std::tuple<RgtSeed, RgtSeed, std::size_t>>& seedsChangeList);
    // };

	// void clearSeedsRepresentation();
	// void addSeedsRepresentation();

	// void clearCurveSpeedupCache();
	// void moveTrackingReference(QPoint pt);
	// void applyDeltaTau(int dTau);
	// void applyDeltaTauRelative(double dTau, std::size_t indexLayerBottom, const std::vector<ReferenceDuo>& referenceVec,
	//			const std::vector<int>& referenceValues);
	// void applyDTauToData();
	// void initSliceReps();

	// void disconnectMoveSlot(QObject* requestingObj);

	// emiting rep is supposed to take care of update itself
	// void connectMoveSlotAndUpdate(QObject* requestingObj);
	// void correctSeedsFromImage(); // only needed image change or data signal endApplyDTauTransaction

	// int getValueFromSliceRepForSeed(SliceRep* rep, std::size_t x, std::size_t y, std::size_t channel);

	SliceDirection m_dir;
	MultiSeedRgt* m_data = nullptr;
	SliceRep* m_seismic = nullptr;
	SliceRep* m_rgt = nullptr;
	QPair<QVector2D,AffineTransformation> m_sliceRangeAndTransfo;
	unsigned int m_currentSlice;

	MultiSeedRgtSliceLayer* m_layer = nullptr;

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

	ContinuousPressEventFilter m_eventFilterClass;

	//jd
	// void setReferences(std::vector<float*> ref);
	// std::vector<float*> m_reference;

	// locks : mandatory because of SliceRep asynchronous loadSlice
	// mutable QMutex m_polygonMutex;

	BEHAVIORMODE m_oldBehaviorMode = FIXED;
};


#endif

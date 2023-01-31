#ifndef MultiSeedHorizon_H
#define MultiSeedHorizon_H

#include <QObject>
#include "editabledata.h"
#include "itreewidgetitemdecoratorprovider.h"
#include "RgtLayerProcessUtil.h"

#include <memory>
#include <map>

class Seismic3DDataset;
class IGraphicRepFactory;
class FixedLayerFromDataset;
class AbstractGraphicRep;
class TextColorTreeWidgetItemDecorator;
// class RgtVolumicDialog;

enum HORIZONMODE {
	DEFAULT = 0,
	DELTA_T = 1
};

enum BEHAVIORMODE {
	FIXED = 0,
	MOUSETRACKING = 1,
	POINTPICKING = 2
};

enum RGTPICKINGMODE
{
	RGTPICKINGNONE = 0,
	RGTPICKINGOK = 1
};

class MultiSeedHorizon: public EditableData, public ITreeWidgetItemDecoratorProvider {
Q_OBJECT
public:
	MultiSeedHorizon(QString name, WorkingSetManager *workingSet,
			Seismic3DDataset *seismic, int channelSeismic,
			Seismic3DDataset *rgt, int channelRgt, QObject *parent = 0);
	virtual ~MultiSeedHorizon();

	QUuid seismicID() const;
	QUuid rgtID() const;

	Seismic3DDataset* seismic() const {
		return m_seismic;
	}

	int channelSeismic() const {
		return m_channelSeismic;
	}

	Seismic3DDataset* rgt() const {
		return m_rgt;
	}

	int channelRgt() const {
		return m_channelRgt;
	}

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override;

	// ITreeWidgetItemDecoratorProvider
	virtual ITreeWidgetItemDecorator* getTreeWidgetItemDecorator() override;

	HORIZONMODE getHorizonMode() const;
	void setHorizonMode(HORIZONMODE mode);

	BEHAVIORMODE getBehaviorMode() const;
	void setBehaviorMode(BEHAVIORMODE mode);

	RGTPICKINGMODE getRgtPickingMode() const;
	void setRgtPickingMode(RGTPICKINGMODE mode);
	// void setRgtVolumicDialog(RgtVolumicDialog *rgtVolumicDialog);
	// RgtVolumicDialog *getRgtVolumicDialog();


	// seed management
	std::size_t addPoint(RgtSeed seed); // add point, can be used for 3D point
	std::size_t addPointAndSelect(RgtSeed seed);
	int pointCount() const;
	std::size_t getSelectedId() const;
	RgtSeed getSelectedPoint() const; // be sure that a point is selected
	RgtSeed& operator[](const std::size_t& id);
	RgtSeed& operator[](std::size_t&& id);
	const RgtSeed& operator[](const std::size_t& id) const;
	const RgtSeed& operator[](std::size_t&& id) const;
	bool selectPoint(std::size_t id); // if index invalid return false and m_currentSeed = 0;
	bool removePoint(std::size_t id);
	void moveSelectedSeed(RgtSeed newSeed); // move point and update
	void moveSeed(std::size_t id, RgtSeed newSeed); // move point and update

	std::vector<std::size_t> getAllIds() const;
	const std::map<std::size_t, RgtSeed>& getMap() const;

	// jd
//	const std::vector<std::vector<RgtSeed>>& getMap2() const;

	void setSeeds(const std::vector<RgtSeed>& seeds);

	// jd
//	void setSeeds2(const std::vector<std::vector<RgtSeed>>& seeds);
	void setReferences(const std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceData);
	std::vector<std::shared_ptr<FixedLayerFromDataset>> getReferences();


	//int getPseudoTau();
	//void setPseudoTau(int);
	bool useSnap() const;
	void toggleSnap(bool);
	int getSnapWindow() const;
	void setSnapWindow(int);
	bool useMedian() const;
	void toggleMedian(bool);
	int getPolarity() const;
	void setPolarity(int);
	int getDistancePower() const;
	void setDistancePower(int);

	int getLWXMedianFilter() const;
	void setLWXMedianFilter(int);

	void setDelta(int d); // give symetric deltas for top and bottom, ex: bottom <- abs(delta); top <- -abs(delta)
	int getDeltaTop() const;
	void setDeltaTop(int d);
	int getDeltaBottom() const;
	void setDeltaBottom(int d);

	int getPseudoTau();
	long getDTauReference();
	void setDTauReference(long dtau);

    void setPseudoTau(int newTau);

    long getDTauPolygonMainCache() const;
    void setDTauPolygonMainCache(long dtau);
    long getDTauRelativePolygonMainCache() const;
    void setDTauRelativePolygonMainCache(long dtau);

    void applyDtauToSeeds(QObject* requestingObj=nullptr);
    void updateSeedsPositionWithRead(QObject* requestingObj);

    bool isMultiSeed() const;
    void setMultiSeed(bool isMultiSeed);

	//void applyDTauToSeeds();
	//void applyRelativeDTauToSeeds();

	//data::StorelessLayer* getConstrainLayer();
	//void setConstrainLayer(data::StorelessLayer* layer);
	//std::vector<std::shared_ptr<data::StorelessLayer>> getReferenceLayer();
	//void setReferenceLayer(const std::vector<std::shared_ptr<data::StorelessLayer>>& layer);

	static std::size_t INVALID_ID;

	void setConstrainLayer(FixedLayerFromDataset* layer);
	FixedLayerFromDataset* constrainLayer();

	bool changeRgtVolume(Seismic3DDataset* newRgt, int newRgtChannel);
	bool changeSeismicVolume(Seismic3DDataset* newSeismic, int newSeismicChannel);

signals:
	void newPointCreated(RgtSeed seed, int id);
	void pointRemoved(RgtSeed seed, int id);
	void pointMoved(RgtSeed oldSeed, RgtSeed newSeed, int id);
	void seedsReset();
	void parametersUpdated();
	void horizonModeChanged(HORIZONMODE mode);
	void behaviorModeChanged(BEHAVIORMODE mode);
	void useSnapChanged(bool useSnap);
	void snapWindowChanged(int snapWindow);
	void useMedianChanged(bool useMedian);
	void distancePowerChanged(int distancePower);
	void lwxMedianFilterChanged(int lwx);
	void polarityChanged(int polarity);
	void deltaChanged(int deltaTop, int deltaBottom);
	void dtauReferenceChanged(long dtauReference);
	void dtauPolygonMainCacheChanged(long dtau);
	void constrainChanged();
	void constrainIsoChanged();
	void referencesChanged();

	// are supposed to be emited by MultiSeedSliceRep to applyDtau only once without other reps interferencies
	void startApplyDTauTransaction(QObject* requestingObj);
	void endApplyDTauTransaction(QObject* requestingObj);

	void rgtChanged();
	void seismicChanged();

private:
	template<typename InputType>
	struct UpdateSeedsPositionWithReadKernel {
		static void run(MultiSeedHorizon* obj, std::vector<std::tuple<RgtSeed, RgtSeed, std::size_t>>& vect);
	};

	template<typename InputType>
	struct UpdateSeedsRgtValueWithDatasetKernel {
		static void run(std::map<std::size_t, RgtSeed>& seeds, Seismic3DDataset* rgt, int channelRgt);
	};

	template<typename InputType>
	struct UpdateSeedsSeismicValueWithDatasetKernel {
		static void run(std::map<std::size_t, RgtSeed>& seeds, Seismic3DDataset* seismic, int channelSeismic);
	};

	std::map<std::size_t, RgtSeed> getSeedsRgtValueWithDataset(const std::map<std::size_t, RgtSeed>& oldSeeds,
			Seismic3DDataset* rgt, int channelRgt);

	std::map<std::size_t, RgtSeed> getSeedsSeismicValueWithDataset(const std::map<std::size_t, RgtSeed>& oldSeeds,
				Seismic3DDataset* seismic, int channelSeismic);

	void constrainBufferUpdated(QString name);

	std::map<std::size_t, RgtSeed> m_seeds;
//	std::vector<std::vector<RgtSeed>> m_seeds2;
	std::size_t m_currentSeed = 0;
	std::size_t m_nextId = 1;

	HORIZONMODE m_horizonMode = DEFAULT;
	BEHAVIORMODE m_behaviorMode = FIXED;
	RGTPICKINGMODE m_rgtPickingMode = RGTPICKINGNONE;

	int m_distancePower = 8;

	int m_polarity = 0;

	bool m_useSnap = false;
	int m_snapWindow = 3;
	int m_lwx = 11;
	bool m_useMedian = false;
	int m_deltaTop = 0;
	int m_deltaBottom = 0;
	long m_dtauPolygonMainCache = 0;
	double m_dtauRelativePolygonMainCache = 0;
	bool m_isMultiSeed = true;

	//data::StorelessLayer* m_constrainLayer = nullptr;
	//std::vector<std::shared_ptr<data::StorelessLayer>> m_referenceLayers;
	long m_dtauReference = 0;
	QString m_name;
	QUuid m_uuid;

	IGraphicRepFactory *m_repFactory;
	Seismic3DDataset *m_seismic;
	int m_channelSeismic;
	Seismic3DDataset *m_rgt;
	int m_channelRgt;
	FixedLayerFromDataset* m_constrainLayer = nullptr;

	// jd
	std::vector<std::shared_ptr<FixedLayerFromDataset>> m_references;
	// RgtVolumicDialog *m_rgtVolumicDialog;

	TextColorTreeWidgetItemDecorator* m_decorator;
};

Q_DECLARE_METATYPE(MultiSeedHorizon*)
#endif

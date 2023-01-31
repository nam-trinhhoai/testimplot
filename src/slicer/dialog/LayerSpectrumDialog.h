#ifndef NEXTVISION_SPECTRALDECOMPDIALOG_H_
#define NEXTVISION_SPECTRALDECOMPDIALOG_H_
#include <QDialog>

#include <QThread>
#include <QSpinBox>
#include <QTimerEvent>
#include <QEvent>
#include <QKeyEvent>
#include <QTimer>
#include <QProgressBar>
#include <QList>

#include <memory>
#include <map>

#include "RgtLayerProcessUtil.h"
#include "multiseedhorizon.h"
#include "SeismicPropagator.h"
#include "fixedlayerfromdataset.h"
#include "continuouspresseventfilter.h"
// #include "RgtVolumicDialog.h"

class QComboBox;
class QCheckBox;
class QDoubleSpinBox;
class QGroupBox;
class QPushButton;
class QListWidget;
class QTreeWidget;
class QProgressBar;
class QLabel;
class QGraphicsScene;
class QTreeWidgetItem;
class QFormLayout;


class OutlinedQLineEdit;
class CollapsableWidget;
class BaseMapGraphicsView;
class GeotimeGraphicsView;
class WindowWidgetPoper2;
class Seismic3DAbstractDataset;
class Seismic3DDataset;
class BaseMapView;
class AbstractSectionView;
class LayerSlice;
class RGBLayerSlice;
class MultiSeedSliceRep;
class Abstract2DInnerView;
class FreeHorizon;
class ItemListSelectorWithColor;
class WellBore;

class MyThread_LayerSpectrumDialogCompute;
class IJKHorizon;

class RgtGraphLabelRead;


class LayerSpectrumDialogSpinBox : public QSpinBox {
	void timerEvent(QTimerEvent *event) {
		event->accept();
	}
};

//class LayerSpectrumTrackerExtension : public QObject/*, public view2d::SyncMultiView2dExtension*/ {
//	Q_OBJECT
//public:
//	LayerSpectrumTrackerExtension();
//	~LayerSpectrumTrackerExtension();
//
////	virtual void setDirection(view2d::View2dDirection dir) override;
////
////	virtual void connectViewer(view2d::SyncViewer2d* viewer) override;
////	virtual void disconnectViewer(view2d::SyncViewer2d* viewer) override;
////
////	virtual void hoverMovedFromSynchro(QPoint imagePoint, PointF2DWithUnit scenePoint, view2d::SyncViewer2d* originViewer) override;
////	virtual void leftButtonPressFromSynchro(QPoint imagePoint, PointF2DWithUnit scenePoint, view2d::SyncViewer2d* originViewer) override;
//
//signals:
////	void hoverMovedFromSynchroSignal(QPoint imagePoint, PointF2DWithUnit scenePoint, view2d::SyncViewer2d* originViewer);
////	void leftButtonPressSignal(QPoint imagePoint, PointF2DWithUnit scenePoint, view2d::SyncViewer2d* originViewer);
//	void keyPressSignal(int key);
//	void keyReleaseSignal(int key);
////	void directionChanged(view2d::View2dDirection dir);
//	//void keyReleaseSignal(int key);
//
//protected:
//	virtual bool eventFilter(QObject* object, QEvent* event) override;
//
//private:
//	QMap<int, std::shared_ptr<QTimer>> filterReleaseKeyMap;
//};
typedef enum {
	eUpdateDatasetS,
	eUpdateDatasetT,
}eUpdateDataSet;

class LayerSpectrumDialog: public QWidget/*, public view2d::SyncMultiView2dExtension */ {
	Q_OBJECT
public:
	friend MyThread_LayerSpectrumDialogCompute;

	typedef struct SpectrumValue {
		int windowSize;
		float hatPower;
		std::vector<RgtSeed> seeds;
		int distancePower;
		bool polarity;
		int halfLwx;
		bool useSnap;
		bool useMedian;
		long id;
	} SpectrumValue;

	typedef struct MorletValue {
		std::vector<RgtSeed> seeds;
		int distancePower;
		bool polarity;
		int halfLwx;
		bool useSnap;
		bool useMedian;
		int freqMin;
		int freqMax;
		int freqStep;
		long id;
	} MorletValue;

	typedef struct GccValue {
		std::vector<RgtSeed> seeds;
		int distancePower;
		bool polarity;
		int halfLwx;
		bool useSnap;
		bool useMedian;
		int offset;
		int w;
		int shift;
		long id;
	} GccValue;

	typedef struct TmapValue {
		std::vector<RgtSeed> seeds;
		int distancePower;
		bool polarity;
		int halfLwx;
		bool useSnap;
		bool useMedian;
		int tmapExampleSize;
		int tmapSize;
		int tmapExampleStep;
		long id;
	} TmapValue;

	typedef struct MeanValue {
		std::vector<RgtSeed> seeds;
		int distancePower;
		bool polarity;
		int halfLwx;
		bool useSnap;
		bool useMedian;
		int meanWindowSize;
		long id;
	} MeanValue;

	typedef struct AnisotropyValue {
		std::vector<RgtSeed> seeds;
		int distancePower;
		bool polarity;
		int halfLwx;
		bool useSnap;
		bool useMedian;
		long id;
	} AnisotropyValue;

	LayerSpectrumDialog(
			Seismic3DAbstractDataset *datasetS,	int channelS, Seismic3DAbstractDataset *datasetT,
			int channelT, GeotimeGraphicsView* viewerMain, Abstract2DInnerView* emitingView,QWidget *parent = 0);
	virtual ~LayerSpectrumDialog();

	void createLayerSlice(eUpdateDataSet eValue = eUpdateDatasetS);
	void removeLayer(Seismic3DAbstractDataset *pSeismic);
	void createLayer();
	void setGeoTimeView(GeotimeGraphicsView*);
	void setSeeds(const std::vector<RgtSeed>& seeds);
	void setPolarity(int pol);
	//void setGeologicalTime(int tau, bool polarity);
	void setPseudoGeologicalTime(int tau); // reset polarity if tau change
	//virtual void hoverMovedFromSynchro(QPoint imagePoint, PointF2DWithUnit scenePoint, view2d::SyncViewer2d* originViewer) override;
	//virtual void leftButtonPressFromSynchro(QPoint imagePoint, PointF2DWithUnit scenePoint, view2d::SyncViewer2d* originViewer) override;
	void methodChanged(int newIndex);

	QWidget* initWidget();
	void updateData(bool fromComputeButton=false);
	void setLayerData();
	void setSynchroSlice(int slice);

	void clearHorizon();
	void setPoint(Abstract2DInnerView* inner2DView, QPointF pt); // clear horizon and add a single point pt

	void set_progressbar_values(double val, double valmax);
	Seismic3DAbstractDataset *getDataSetS();
	LayerSlice* getMdata();
	std::vector<RgtSeed> getSeeds();
	std::unique_ptr<FixedLayerFromDataset> getMConstrainData();
	float *getHorizonFromSeed();
	void eraseParentPatch(Abstract2DInnerView* inner2DView, QPointF pt);

private:
//	model::DatasetGroup* getOrCreateDatasetSpectrum( model::Dataset* inputDataset,
//			std::string newGroupName, std::string newDatasetName, int nbSlices);
//	void connectViewer(view2d::SyncViewer2d* viewer);
//	void disconnectViewer(view2d::SyncViewer2d* viewer);
	template<typename DataType> void SetDataItem(IData *pData,std::size_t index);
	template<typename DataType> std::size_t getViewIndex(QString titleName);
	template<typename DataType> std::size_t getViewIndexOrCreate(QString titleName);
	void crunch();
	void threadableCompute();
	void trt_compute();
	void postComputeStep();

	void setupHorizonExtension();
	void toggleDebugMode(bool state);
	void fillHistorySpectrum();
	void appendHistorySpectrum(SpectrumValue val);
	void appendHistorySpectrum();
	void fillHistoryMorlet();
	void appendHistoryMorlet(MorletValue val);
	void appendHistoryMorlet(void);
	void fillHistoryGcc(void);
	void appendHistoryGcc(GccValue val);
	void appendHistoryGcc(void);
	void fillHistoryTmap();
	void appendHistoryTmap(TmapValue val);
	void appendHistoryTmap(void);
	void fillHistoryMean();
	void appendHistoryMean(MeanValue val);
	void appendHistoryMean(void);
	void fillHistoryAnisotropy();
	void appendHistoryAnisotropy(AnisotropyValue val);
	void appendHistoryAnisotropy(void);
	void tryAppendHistory();
	//void computeModuleDataset();

	void undoHorizonModification();
	void redoHorizonModification();
	void loadHorizon();
	void saveHorizon();
	void saveFreeHorizon(QString rgtPath, QString filename, float *data, int dimy, int dimz, float tdeb, float pasech);

    void propagateHorizon();
    void propagationCacheSetup();
    void undoPropagation();
    void updatePropagationView();
    void updatePropagationExtension();

    void cloneAndKeep();
    void eraseData();

	void toggleSeedMode(bool isMultiSeed);
	void interpolateHorizon(bool toggled);
	void initReferenceComboBox();
	bool filterHorizon(FreeHorizon* horizon);
	bool filterHorizon(IJKHorizon* horizon);
	// 0->seed 1->neightbour
	void patchConstraintType(int type, std::vector<int> &vy, std::vector<int> &vz);
	void patchConstrain();
	void patchNeightbourConstrain();
	void updatePatchDataSet();

	void newPointCreatedFromHorizon(RgtSeed seed, int id);
	bool saveRGB(QString saveRGBName);

	long getMeanTauFromMultiSeedHorizon() const;

	// pair second output is the found data, first output is true if the data is in the expected working set manager
	std::pair<bool, LayerSlice*> searchFirstValidDataFromMethod(int method);

	void reloadHorizonList();

	void clearLayerSlices();
	void fullClearHorizon(); // do clearHorizon + destroy m_horizon, constrain and propagator
	void addDataset(Seismic3DAbstractDataset* dataset);

	static std::vector<Seismic3DAbstractDataset*> getSelectedDatasetsInView(GeotimeGraphicsView* view);

	void createConstrain();
	void releaseConstrainFromDestroyed(); // this should only be called by the QObject::destroyed signal of the constrain

	std::size_t nextDatasetIndex() const;
	mutable std::size_t m_nextDatasetIndex = 0;

	//QWidget *createFilteringParameterWidget();
	//FilterApplier createFilterApplier() override;

	GeotimeGraphicsView* m_originViewer;
	Abstract2DInnerView* m_emitingView;
	Seismic3DAbstractDataset *m_datasetS;
	int m_channelS;
	Seismic3DAbstractDataset *m_datasetT;
	int m_channelT;
	int m_method = 1;
	std::vector<RgtSeed> m_seeds;
	std::vector<RgtSeed> m_patchSeeds;
	int m_patchCurrentPolarity = 0;
	std::vector<unsigned int> m_patchSeedId;
	std::vector<float> m_patchTabIso;
	int m_distancePower = 8;
	int m_pseudoTau = 0;
	int m_tauStep = 100;
	int m_polarity = 0; // -1 for negative, 0 for not defined, 1 for positive
	int m_halfLwxMedianFilter = 11;
	bool m_debugMode = false;
	bool m_useSnap = false;
	int m_snapWindow = 3;
	bool m_useMedian = false;
	bool m_computeButtonOnly = false;
	int m_spectrumWindowSize = 64;
	float m_hatPower = 5;

	int m_freqMin = 30;
	int m_freqMax = 150;
	int m_freqStep = 5;

	int m_gccOffset = 7;
	int m_w = 7;
	int m_shift = 5;

	int m_tmapExampleSize = 10;
	int m_tmapSize = 33;
	int m_tmapExampleStep = 20;

	int m_meanWindowSize = 15;
	QList<std::pair<Seismic3DDataset*, int>> m_meanDatasets;
	std::map<std::size_t, Seismic3DDataset*> m_allDatasets;

	long currentId = 0;

	std::vector<SpectrumValue> m_historySpectrum;
	std::vector<MorletValue> m_historyMorlet;
	std::vector<GccValue> m_historyGcc;
	std::vector<TmapValue> m_historyTmap;
	std::vector<MeanValue> m_historyMean;
	std::vector<AnisotropyValue> m_historyAnisotropy;
//	view2d::View2dDirection m_direction;
//	int m_slice;
	RgtGraphLabelRead *rgtGraphLabelRead = nullptr;

	// ui
	QPushButton* m_seedModeButton = nullptr;

	// multi seed controls
	QPushButton* m_seedEditButton = nullptr;
	QPushButton* m_undoButton = nullptr;
	QPushButton* m_redoButton = nullptr;
	QPushButton* m_loadHorizonButton = nullptr;
	QPushButton* m_saveButton = nullptr;
	QPushButton* m_releaseButton = nullptr;

	// mono seed controls
	LayerSpectrumDialogSpinBox* m_tauSpinBox = nullptr;
	QWidget* m_tauHolder = nullptr;
	QLabel* m_tauHolderLabel = nullptr;
	QSpinBox* m_tauStepSpinBox = nullptr;
	QLabel* m_tauStepLabel = nullptr;

	QLabel* m_polarityLabel = nullptr;
	QComboBox* m_polarityComboBox = nullptr;
	QSpinBox* m_distancePowerSpinBox = nullptr;
	QLabel* m_distancePowerLabel = nullptr;
	QLabel* m_halfLwxMedianFilterLabel = nullptr;
	QSpinBox* m_halfLwxMedianFilterSpinBox = nullptr;
	QCheckBox* m_useSnapCheckBox = nullptr;
	QLabel* m_useSnapLabel = nullptr;
	QSpinBox* m_snapWindowSpinBox = nullptr;
	QLabel* m_snapWindowLabel = nullptr;
	QCheckBox* m_useMedianCheckBox = nullptr;
	QLabel* m_useMedianLabel = nullptr;
	QPushButton* m_computeButton = nullptr;
#if 1
	QComboBox* m_displayViewCombo = nullptr;
#endif
	QComboBox* m_methodCombo = nullptr;
	QListWidget * m_rgtCombo = nullptr;
	QListWidget *m_sismiqueCombo = nullptr;
	QListWidget* m_patchList = nullptr;
	QListWidget* m_historySpectrumList = nullptr;
	QListWidget* m_historyMorletList = nullptr;
	QListWidget* m_historyGccList = nullptr;
	QListWidget* m_historyTmapList = nullptr;
	QListWidget* m_historyMeanList = nullptr;
	QListWidget* m_historyAnisotropyList = nullptr;
	QProgressBar* m_progressbar = nullptr;
	QCheckBox* m_debugModeCheckBox = nullptr;
	QCheckBox* m_computeButtonOnlyCheckBox = nullptr;

	QPushButton* m_propagateButton = nullptr;
	QPushButton* m_undoPropagationButton = nullptr;
	QPushButton* m_undoPatchButton = nullptr;
	QPushButton* m_toggleInterpolation = nullptr;
	QPushButton* m_cloneAndKeepButton = nullptr;
	QPushButton* m_eraseDataButton = nullptr;
	QSpinBox* m_sizeCorrSpinBox = nullptr;
	QLabel* m_sizeCorrLabel = nullptr;
	QDoubleSpinBox* m_seuilCorrSpinBox = nullptr;
	QLabel* m_seuilCorrLabel = nullptr;
	QSpinBox* m_seedFilterNumberSpinBox = nullptr;
	QLabel* m_seedFilterNumberLabel= nullptr;
	QComboBox* m_propagationTypeComboBox = nullptr;
	QLabel* m_propagationTypeLabel = nullptr;
	QSpinBox* m_numIterSpinBox = nullptr;
	QLabel* m_numIterLabel = nullptr;

	QFormLayout* m_commonForm = nullptr;

	// LayerSpectrum
	QGroupBox* m_spectrumGroupBox = nullptr;
	QSpinBox* m_spectrumWindowSizeSpinBox = nullptr;
	QDoubleSpinBox* m_spectrumHatPower = nullptr;
	QLabel* m_spectrumHatPowerLabel = nullptr;

	// Morlet
	QGroupBox* m_morletGroupBox = nullptr;
	QSpinBox* m_minFreqSpinBox = nullptr;
	QSpinBox* m_maxFreqSpinBox = nullptr;
	QSpinBox* m_stepFreqSpinBox = nullptr;

	// Gcc
	QGroupBox* m_gccGroupBox = nullptr;
	QSpinBox* m_gccOffsetSpinBox = nullptr;
	QSpinBox* m_wSpinBox = nullptr;
	QSpinBox* m_shiftSpinBox = nullptr;
	
	// Tmap
	QGroupBox* m_tmapGroupBox = nullptr;
	QSpinBox* m_tmapExampleSizeSpinBox = nullptr;
	QSpinBox* m_tmapSizeSpinBox = nullptr;
	QSpinBox* m_tmapExampleStepSpinBox = nullptr;

	// Mean
	QGroupBox* m_meanGroupBox = nullptr;
	QSpinBox* m_meanWindowSizeSpinBox = nullptr;
	QListWidget* m_meanDatasetListWidget = nullptr;

	// Anisotropy
	QGroupBox* m_anisotropyGroupBox = nullptr;
	QTreeWidget* m_anisotropyTree = nullptr;

	MultiSeedHorizon* m_horizon = nullptr;
//	MultiSeedSliceRep* m_horizonRep = nullptr; // m_horizon rep on m_originViewer
	//PointPickingTask m_pickingTaskSection;
	//PointPickingTask m_pickingTaskMap;

	// linked window
	WindowWidgetPoper2* m_window = nullptr;
	BaseMapGraphicsView* m_datasetView = nullptr;
	//view2d::SyncMultiView2d* m_syncView = nullptr;
	//PropertiesRegister m_parameters;

    std::vector<unsigned int> m_propagatorUndoCache;
    unsigned int m_propagatorNextId = 1;
    std::vector<long> m_propagatorNewSeedsIdLimit; // length should be m_propagatorNextId - 1
    // propagate parameters
    int m_sizeCorr = 10; // halfWindow
    float m_seuilCorr = 0.95;
    //int m_seedReductionWindow = 10; // half window
    int m_seedFilterNumber = 1000;
    int m_propagationType = 1;
    bool m_isDataInterPolated = false;
    int m_numIter = 2;



	bool m_isComputationValid = false;
	bool m_isWindowChanged = true;
	//bool m_isMultiSeedActive = true;
	bool m_seedEdit = true;
	bool m_isConstrainInitialized = false;
	std::vector<RgtSeed> m_redoButtonList;

	ContinuousPressEventFilter m_geotimeViewTrackerExtension;
	bool m_isDataInterpolated = false;

	QProgressBar *qpb_progress1;
	MyThread_LayerSpectrumDialogCompute *pthread = nullptr;

	double progressbar_val = 0.0;
	double progressbar_valmax = 1.0;

	ItemListSelectorWithColor *m_referenceList;
//	std::vector<float*> m_referenceData;

private:
	QVector<IData*> m_VectorData;
	QMap<Seismic3DAbstractDataset*,QVector<IData*> > m_mapData;
	LayerSlice* m_data = nullptr;
	RGBLayerSlice* m_dataRGB = nullptr;
	std::unique_ptr<FixedLayerFromDataset> m_constrainData = nullptr;
//	std::unique_ptr<FixedLayerFromDataset> m_constrainDataIsochrone = nullptr;
	std::vector<std::shared_ptr<FixedLayerFromDataset>> m_referenceLayers;
	std::unique_ptr<SeedsGenericPropagator> m_propagator;
	Seismic3DDataset* m_cachePatchDataset = nullptr;
	bool m_constrainMissingWarned = false;

	std::map<long, QString> m_horizonNames;
	std::map<long, QString> m_horizonPaths;
	std::map<long, FreeHorizon*> m_horizonDatas;

	bool m_isComputationRunning = false;

	void loadReferenceHorizon(QTreeWidgetItem* item);
	void unloadReferenceHorizon(std::size_t index_ref);
	void UpdateDatasets(Seismic3DAbstractDataset *pDataSet);

	// RgtVolumicDialog * m_rgtVolumicDialog = nullptr;
	QList<QMetaObject::Connection> m_conn;
	QLineEdit *m_patchThreshold = nullptr;

	long m_horizonNextId = 0;


private slots:
	void updateReferenceHorizonList();
	void updateReferenceHorizonColor(QTreeWidgetItem* item, QColor color);
// 	void trt_compute();
	void showTime();	
	void trt_rgtModifications();
	void newFreeHorizonAdded(IData *data);
	void newLayerAdded(IData *data);
	void updateDataSet();
//	void updateDataSetT();
	void updateRgtList(AbstractGraphicRep *rep);
	void choosedPicksFromQC(QList<WellBore*> choosenWells, QList<int> geotime, QList<int> mds);
	void updateAngleTree();
	void anisotropyDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
	void erasePatch();
};

class MyThread_LayerSpectrumDialogCompute : public QThread
{
	public:		
		LayerSpectrumDialog *pspectrum_dialog = nullptr;		
     // Q_OBJECT
	 /*
	 public:
	 MyThread_file_convertion();
	 QString src_filename, dst_filename;
	 float cwt_error;
     int cpt, cpt_max, cont, complete, abort;
	 private:
	 // GeotimeConfigurationWidget *pp;
	 */
protected:
     void run();
};

#endif /* NEXTVISION_SPECTRALDECOMPDIALOG_H_ */

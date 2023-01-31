#ifndef GeotimeGraphicsView_H
#define GeotimeGraphicsView_H

#include "multitypegraphicsview.h"
#include "viewutils.h"
#include "propertypanel.h"
#include "tools3dWidget.h"
#include "nurbswidget.h"
#include "multiview.h"

#include <QList>
#include <QVector3D>

#include <memory>
#include <QMutex>
#include <QPoint>

class QInnerViewTreeWidgetItem;
class AbstractInnerView;
class Abstract2DInnerView;
class LayerSpectrumDialog;
class RgtVolumicDialog;
class QTreeWidget;
class QToolButton;
class Seismic3DAbstractDataset;
class SeismicSurvey;
class DataSelectorDialog;
class StackSynchronizer;
class Tools3dWidget;
class NurbsWidget;
class MtLengthUnit;
class ProcessRelay;
class IComputationOperator;
//class OrderStackHorizonWidget;

//class MultiView;

class QComboBox;
class QTreeWidgetItem;
class IData;


enum TemplateOperation {
	SplitHorizontal, SplitVertical, AddAsTab
};

typedef struct TemplateInnerView {
	long viewId = -1;
	long targetId = -1;
	ViewType viewType;
	TemplateOperation operation = SplitHorizontal;
	QString title = "";
} TemplateInnerView;

typedef QList<TemplateInnerView> TemplateView;

class GeotimeGraphicsView: public MultiTypeGraphicsView {
Q_OBJECT
public:
	GeotimeGraphicsView(WorkingSetManager *factory, QString uniqueName, QWidget *parent);
	virtual ~GeotimeGraphicsView();

	void setTemplate(const TemplateView&);
	QInnerViewTreeWidgetItem* getItemFromView(AbstractInnerView* view);
	QVector<AbstractInnerView *> getInnerViews() const; // public version of protected innerViews from AbstractGeaphicsView

	void setDataSelectorDialog(DataSelectorDialog* dialog);

	const MtLengthUnit* depthLengthUnit() const;
	void setDepthLengthUnit(const MtLengthUnit* unit);
	void toggleDepthLengthUnit();

	// does not take ownership of the given relay
	void setProcessRelay(ProcessRelay* relay);

	static std::vector<AbstractInnerView*> getInnerViewsList();
	static WorkingSetManager* getWorkingSetManager();
	void SetDataItem(IData *pData,std::size_t index,Qt::CheckState state );
	AbstractInnerView* getInnerView3D(int index);




signals:
	void depthLengthUnitChanged(const MtLengthUnit* depthUnit);
	void registerAddView3D();

protected slots:
	void viewPortChanged(const QPolygonF &poly);
	void zScaleChanged(double zScale);
	void positionCamChanged(QVector3D position);
	void viewCenterCamChanged(QVector3D center);
	void upVectorCamChanged(QVector3D up);
	void resetInnerViewsPositions();
	virtual void unregisterView(AbstractInnerView *toBeDeleted) override;
	void contextMenu(Abstract2DInnerView* emitingView, double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& mainMenu);
//	void addDataFromComputation();

	void openSeismicInformation();
	void openHorizonInformation();
	void openIsoHorizonInformation();
	void openWellsInformation();
	void openPicksInformation();
	void openNurbsInformation();
	void openHorizonAnimationInformation();
	void openImportSismage();

protected:
	void clearInnerViews();
	virtual void registerView(AbstractInnerView *newView, TemplateOperation operation, KDDockWidgets::DockWidget* operationTarget);
	virtual void registerView(AbstractInnerView *newView) override;
	void closeEvent(QCloseEvent *event);

	// return true if dataset data has been added
	bool initSection(AbstractInnerView* newView);
	bool initSectionFromScratch(AbstractInnerView* newView);
	bool initSectionFromPrevious(AbstractInnerView* newView, AbstractInnerView* reference);

private slots:
	void freeHorizonChangeColor(QTreeWidgetItem *item);

	void openDataManager();
	void openGraphicToolsDialog();
	void processDataAddedRelay(long id, IComputationOperator* obj);
//	void createAnimatedSurfacesImages();
//	void createAnimatedSurfacesInit();
//	void createAnimatedSurfacesCube();
//	void createAnimatedSurfacesCubeRgb1();
//	void createAnimatedSurfacesCubeMean();
//	void createAnimatedSurfacesCubeGcc();

//	void createVideoLayer();
//	void createTmapLayer();
//	void createRGBVolumeFromCubeAndXt();
	void manageSynchro();

	void showProperties();
	void showMultiView();

	void saveSession();
	void simplifySurface(int);
	void simplifySeuilLogs(int);
	void simplifySeuilWell(double);
	void showInfo3D(bool);
	void showGizmo3D(bool);
	void setspeedUpDown(float);
	void setspeedHelico(float);
	void setspeedRotHelico(float);

	void showNormalsWell(bool);
	void wireframeWell(bool);


	void setDiameterPick(int);
	void setThicknessPick(int);
	void setThicknessLog(int);
	void setColorLog(QColor);
	void setColorWell(QColor);
	void setColorSelectedWell(QColor);
	void setDiameterWell(int);
	void setWellMapWidth(double);
	void setWellSectionWidth(double);

	void setSpeedAnim(int);
	void setAltitudeAnim(int);

	void show3dTools(bool);

	void showHelico(bool);

	void computeReflectivity();



private:
	void spectrumDecomposition(Abstract2DInnerView* emitingView, double worldX, double worldY);
	void createSpectrumData(Abstract2DInnerView* emitingView);
	void erasePatch(Abstract2DInnerView* emitingView, double worldX, double worldY);
	void createGenericTreatmentData(Abstract2DInnerView* emitingView);
	void copyRandom(Abstract2DInnerView* emitingView);
	void rgtModification(Abstract2DInnerView* emitingView, double worldX, double worldY);
	Seismic3DAbstractDataset *getSeismic3DAbstractDataset();
	QTreeWidgetItem * getItemFromTreeWidget(QTreeWidgetItem *item0, QString name);





	/**
	 * Selection function
	 * In will select the items from the data in all the following views :
	 * AbstractSectionViews, RandomLineview, ViewQt3D and StackBasemapView named "RGB"
	 */
	void selectLayerTypedData(IData* data);

	Seismic3DAbstractDataset* getDataset(bool onlyCpu);
	SeismicSurvey* getSurvey();

	std::size_t m_inlineCount = 0;
	std::size_t m_xlineCount = 0;
	std::size_t m_randomCount = 0;
	std::size_t m_basemapCount = 0;
	std::size_t m_view3dCount = 0;
	std::size_t m_otherViewCount = 0;

	LayerSpectrumDialog* m_layerSpectrumDialog = nullptr;
	RgtVolumicDialog *m_rgtVolumicDialog = nullptr;
	QToolButton* m_depthUnitButton;

	DataSelectorDialog* m_dataSelectorDialog = nullptr;

	PropertyPanel* m_properties = nullptr;
	Tools3dWidget* m_tools3D = nullptr;

	//NurbsWidget* m_nurbs3D = nullptr;

	// for ViewQt3D
	double m_zScale = 1;
	QVector3D m_posCam;
	QVector3D m_viewCam;
	QVector3D m_upCam;

	std::unique_ptr<StackSynchronizer> m_stackSynchronizer;
	static std::vector<AbstractInnerView*> m_InnerViewVec;
	static WorkingSetManager* m_WorkingSetManager;
        QMutex m_mutexView;

	const MtLengthUnit* m_depthLengthUnit = nullptr;

	ProcessRelay* m_processRelay = nullptr;

	//OrderStackHorizonWidget* m_orderStackHorizon = nullptr;

	//MultiView* m_multiView = nullptr;
};

#endif

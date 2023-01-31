#ifndef NEXTVISION_RGTVOLUMICDIALOG_H_
#define NEXTVISION_RGTVOLUMICDIALOG_H_
#include <QDialog>

#include <QThread>
#include <QSpinBox>
#include <QTimerEvent>
#include <QEvent>
#include <QKeyEvent>
#include <QTimer>
#include <QMap>
#include <QProgressBar>
#include <vector>

#include <memory>

#include "RgtLayerProcessUtil.h"
#include "multiseedrgt.h"
#include "pointpickingtask.h"
#include "SeismicPropagator.h"
#include "fixedlayerfromdataset.h"
#include "continuouspresseventfilter.h"
#include "multiseedslicerep.h"

class QComboBox;
class QCheckBox;
class QDoubleSpinBox;
class QGroupBox;
class QPushButton;
class QListWidget;
class QProgressBar;
class QLabel;
class QGraphicsScene;
class QListWidgetItem;


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

class MyThread_LayerSpectrumDialogCompute;
class IJKHorizon;

class LayerSpectrumDialog;




class RgtVolumicDialog: public QWidget/*, public view2d::SyncMultiView2dExtension */ {
	Q_OBJECT
public:

	RgtVolumicDialog(
			Seismic3DAbstractDataset *datasetS,	int channelS, Seismic3DAbstractDataset *datasetT, int channelT,
			GeotimeGraphicsView* viewerMain, QWidget *parent = 0);
	RgtVolumicDialog();
	virtual ~RgtVolumicDialog();
	void setMultiSeedHorizon(MultiSeedHorizon *multiseedhorizon);
	void setLayerSpectrumDialog(LayerSpectrumDialog *layerSpectrumDialog);
	void setLayerSpectrumDialog(MultiSeedSliceRep *multiSeedSliceRep);
	void addPoint(RgtSeed seed);

private:
	GeotimeGraphicsView* m_originViewer;
	Seismic3DAbstractDataset *m_datasetS;
	Seismic3DAbstractDataset *m_datasetT;
	PointPickingTask m_pickingTaskMap;
	ContinuousPressEventFilter m_geotimeViewTrackerExtension;

	QListWidget *qListWidgetPicking;

	std::vector<std::vector<RgtSeed>> constraintPoints;
	std::vector<QString> constraintName;

	int m_channelS;
	int m_channelT;
	MultiSeedRgt* m_rgt = nullptr;
	MultiSeedHorizon *m_multiSeedHorizon = nullptr;
	LayerSpectrumDialog *m_layerSpectrumDialog = nullptr;
	MultiSeedSliceRep *m_multiSeedSliceRep = nullptr;
	int constraintCounter;
	void initIhm();
	void displayConstraintNames();
	int getConstraintSelectedIndex();
	void addPoints(std::vector<RgtSeed> seeds);
	std::vector<float*> getConstaintHorizons();
	void rgtRun();

private slots:
	void trt_displayConstraints();
	void trt_startNewPicking();
	void trt_erasePicking();
	void trt_runRgt();
	int trt_constraintListClick(QListWidgetItem* list);
	void trt_debugClick();
};


#endif

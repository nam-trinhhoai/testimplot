#ifndef SRC_WIDGET_BNNI_TRAININGSETPARAMETERWIDGET_H_
#define SRC_WIDGET_BNNI_TRAININGSETPARAMETERWIDGET_H_

#include "bnnitrainingset.h"
#include "WellUtil.h"

#include <QWidget>
#include <QDialog>


class QGridLayout;
class QVBoxLayout;
class QHBoxLayout;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QComboBox;
class QPushButton;
class QTreeWidget;
class QTreeWidgetItem;
class QDialogButtonBox;
class QCheckBox;
class QScrollArea;
class QSpinBox;

class ProjectManagerWidget;
class CustomChartView;

class MtLengthUnit;

//namespace QtCharts {
	class QLineSeries;
	class QChart;
//};

// to separate class in other file later
class WellHeaderCell : public QWidget {
	Q_OBJECT
public:
	WellHeaderCell(BnniTrainingSet* trainingSet, long wellId,
			QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~WellHeaderCell();

	long wellId() const;

public slots:
	void updateName();

private slots:
	void openMenu();
	void askDeleteSlot();
	void askChangeWellSlot();

signals:
	void askDelete();
	void askChangeWell();

private:
	long m_wellId;
	BnniTrainingSet* m_trainingSet;

	QLabel* m_wellLabel;
	QPushButton* m_menuButton;
};

class KindHeaderCell : public QWidget {
	Q_OBJECT
public:
	KindHeaderCell(BnniTrainingSet* trainingSet, long kindId, QWidget *parent = nullptr,
			Qt::WindowFlags f = Qt::WindowFlags());
	~KindHeaderCell();

	long kindId() const;

signals:
	void askDelete();

private slots:
	void changeFilterType(int idx);
	void changeKind();
	void openMenu();
	void askDeleteSlot();
	void changeMin(double min);
	void changeMax(double max);

private:
	BnniTrainingSet* m_trainingSet;
	long m_kindId;
	QComboBox* m_typeComboBox;
	QLineEdit* m_nameLineEdit;
	QDoubleSpinBox* m_minSpinBox;
	QDoubleSpinBox* m_maxSpinBox;
	QPushButton* m_menuButton;

	int m_NAME_INDEX = 0;
	int m_KIND_INDEX = 1;
};

class TfpHeaderCell : public QWidget {
	Q_OBJECT
public:
	TfpHeaderCell(BnniTrainingSet* trainingSet, QWidget *parent = nullptr,
			Qt::WindowFlags f = Qt::WindowFlags());
	~TfpHeaderCell();
private slots:
	void changeName();
private:
	BnniTrainingSet* m_trainingSet;
	QLineEdit* m_nameLineEdit;
};

class WellKindCell: public QWidget {
	Q_OBJECT
public:
	WellKindCell(BnniTrainingSet* trainingSet, long wellId, long kindId,
			QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~WellKindCell();

	long wellId() const;
	long kindId() const;

public slots:
	void updateName();

signals:
	void askChangeLog();

private slots:
	void openMenu();
	void askChangeLogSlot();

private:
	long m_wellId;
	long m_kindId;
	BnniTrainingSet* m_trainingSet;

	QLabel* m_nameLabel;
	QPushButton* m_menuButton;
};

class WellTfpCell : public QWidget {
	Q_OBJECT
public:
	WellTfpCell(BnniTrainingSet* trainingSet, long wellId,
			QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~WellTfpCell();

	long wellId() const;

public slots:
	void updateName();

signals:
	void askChangeTfp();

private slots:
	void askChangeTfpSlot();

private:
	long m_wellId;
	BnniTrainingSet* m_trainingSet;

	QLabel* m_nameLabel;
	QPushButton* m_menuButton;
};

class LogSelectorTreeDialog : public QDialog {
	Q_OBJECT
public:
	LogSelectorTreeDialog(const std::vector<WELLHEADDATA>& wellHead, QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~LogSelectorTreeDialog();

	int wellHeadIdx() const;
	int wellBoreIdx() const;

private slots:
	void tryAccept();
	void treeSelectionChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous);

private:
	void updateAcceptButtons();

	const std::vector<WELLHEADDATA>& m_wellHead;
	int m_wellHeadIdx = -1;
	int m_wellBoreIdx = -1;

	QTreeWidget* m_treeWidget;
	QDialogButtonBox* m_buttonBox;
};

class SeismicCell : public QWidget {
	Q_OBJECT
public:
	SeismicCell(BnniTrainingSet* trainingSet, long seismicId,
			QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~SeismicCell();

	long seismicId() const;

public slots:
	void updateName();
	void changeMin(double min);
	void changeMax(double max);

private slots:
	void openMenu();
	void askDeleteSlot();
	void askChangeSeismicSlot();
	void changeMinInternal(double min);
	void changeMaxInternal(double max);

signals:
	void askDelete();
	void askChangeSeismic();

private:
	long m_seismicId;
	BnniTrainingSet* m_trainingSet;

	QLabel* m_seismicLabel;
	QPushButton* m_menuButton;
	QDoubleSpinBox* m_minSpinBox;
	QDoubleSpinBox* m_maxSpinBox;
};

class AutoCorrControlWidget : public QWidget {
	Q_OBJECT
public:
	AutoCorrControlWidget(const std::vector<QString>& seismicNames, const std::vector<QString>& seismicPaths,
			QWidget* parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~AutoCorrControlWidget();

	void setSampleRate(float newSampleRate);

private:
	void changeCurrentSeismic(int index);
	void computeAutoCorrelation();

	std::vector<QString> m_seismicNames;
	std::vector<QString> m_seismicPaths;
	int m_currentSeismic = -1;
	float m_sampleRate = 0.5;

	QLineSeries* m_seriesAutoCorr = nullptr;
	QChart* m_chartAutoCorr = nullptr;
	CustomChartView* m_chartViewAutoCorr = nullptr;
	QComboBox* m_seismicComboBox = nullptr;

};

class TrainingSetParameterWidget : public QWidget {
	Q_OBJECT
public:
	TrainingSetParameterWidget(QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~TrainingSetParameterWidget();

	static QString getNameFromSampleUnit(SampleUnit unit);

public slots:
	void compute();
	long addNewWell();
	void addNewKind();
	void addNewSeismic();
	void addMoreWells();
	void changeSampleRate(double val);
	void toggleHorizonInterval(int state);
	void selectTopHorizon();
	void selectBottomHorizon();
	void changeTopDelta(double delta);
	void changeBottomDelta(double delta);
	void toggleDepthUnit();
	void trainingSetSeismicUnitChanged();

private:
	void setTrainingSetName();
	void projectChanged();
	void setBandPassState(int state);
	void setBandPassFreq(double freq);
	void setAugmentationState(int state);
	void changeAugmentationDistance(int dist);
	void changeAugmentationNoiseStd(double val);
	void changeUseCnxAugmentation(int state);
	void setTrainingSetNameFromData(QString file);
	void addTfpColumn();

	WellHeaderCell* createWellCell(long wellId);
	KindHeaderCell* createKindCell(long kindId);
	TfpHeaderCell* createTfpCell();
	WellKindCell* createWellKindCell(long wellId, long kindId);
	WellTfpCell* createWellTfpCell(long wellId);
	SeismicCell* createSeismicCell(long seismicId);

	void renameWellKindCell(WellKindCell* cell);
	void renameWellTfpCell(WellTfpCell* cell);
	void deleteKindHeaderCell(KindHeaderCell* headerCell, int columnIndex);
	void deleteWellHeaderCell(WellHeaderCell* headerCell, int lineIndex);
	void changeWellWellHeaderCell(WellHeaderCell* headerCell);
	void changeWellInData(long wellId, const WELLBOREDATA& wellBore,
			int cacheHeadIdx, int cacheBoreIdx);

	void deleteSeismicCell(SeismicCell* cell);
	void changeSeismic(SeismicCell* cell);

	void initLogsLayout();
	void initSeismicsLayout();
	void initLogsMap(BnniTrainingSet::WellParameter& well, const WELLBOREDATA& wellBoreList);
	void initTfpsMap(BnniTrainingSet::WellParameter& well, const WELLBOREDATA& wellBore);

	bool wellMatch(const BnniTrainingSet::WellParameter& wellParameter, const WELLBOREDATA& wellBore);

	QString getKind(QString logPath);

	void openAutoCorrWindow();
	std::vector<long> detectEmptyWells();

	void fillIncompleteWells();

	// params
	BnniTrainingSet m_trainingSet;
	long m_logWellId = 1;
	long m_logKindId = 2;

	// ui
	ProjectManagerWidget* m_projectManagerWidget;

	// ui logs
	QGridLayout* m_logsLayout;
	QCheckBox* m_bandpassCheckBox;
	QLabel* m_bandPassFreqLabel;
	QDoubleSpinBox* m_bandPassFreqSpinBox;
	QCheckBox* m_augmentationCheckBox;
	QLabel* m_augmentationDistanceLabel;
	QSpinBox* m_augmentationDistanceSpinBox;
	QLabel* m_useCnxAugmentationLabel;
	QCheckBox* m_useCnxAugmentationCheckBox;
	QLabel* m_augmentationNoiseStdLabel;
	QDoubleSpinBox* m_augmentationNoiseStdSpinBox;
	QLineEdit* m_trainingSetLineEdit;
	QPushButton* m_addKind;
	QPushButton* m_addWell;

	// ui seismic
	QVBoxLayout* m_seismicsLayout;
	QPushButton* m_addSeismic;

	// ui horizon
	QScrollArea* m_horizonHolder;
	QHBoxLayout* m_horizonLayout;
	QLabel* m_topHorizonLabel;
	QDoubleSpinBox* m_topDeltaSpinBox;
	QLabel* m_bottomHorizonLabel;
	QDoubleSpinBox* m_bottomDeltaSpinBox;

	// ui depth unit
	QPushButton* m_depthUnitButton;

	const MtLengthUnit* m_depthUnit;

	long m_horizonIntervalId = -1;

	std::map<long, WellHeaderCell*> m_wellHeaderCells;
	std::map<long, KindHeaderCell*> m_kindHeaderCells;
	std::map<long, std::map<long, WellKindCell*>> m_wellKindCells;// first id is wellId then it is kindId
	std::map<long, WellTfpCell*> m_wellTfpCells;

	std::vector<AutoCorrControlWidget*> m_displayedAutoCorrWidgets;
	std::map<AutoCorrControlWidget*, QMetaObject::Connection> m_autoCorrConns;

	// cache
	std::map<QString, QString> m_logKindDetectionCache;
};

#endif

#ifndef SRC_WIDGET_COMPUTEREFLECTIVITYWIDGET_H
#define SRC_WIDGET_COMPUTEREFLECTIVITYWIDGET_H

#include <QDialog>
#include <QWidget>

class QComboBox;
class QDialogButtonBox;
class QDoubleSpinBox;
class QGridLayout;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QTreeWidget;
class QTreeWidgetItem;

class ComputeReflectivityWidget;
class FolderData;
class WellBore;

// copied from trainingsetparameterwidget, maybe this principle should be made into a more generic way

// to separate class in other file later
class WellHeaderCellReflectivity : public QWidget {
	Q_OBJECT
public:
	WellHeaderCellReflectivity(ComputeReflectivityWidget* oriWidget, long wellId,
			QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~WellHeaderCellReflectivity();

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
	ComputeReflectivityWidget* m_oriWidget;

	QLabel* m_wellLabel;
	QPushButton* m_menuButton;
};

class KindHeaderCellReflectivity : public QWidget {
	Q_OBJECT
public:
	KindHeaderCellReflectivity(ComputeReflectivityWidget* oriWidget, long kindId, QWidget *parent = nullptr,
			Qt::WindowFlags f = Qt::WindowFlags());
	~KindHeaderCellReflectivity();

	long kindId() const;

private slots:
	void changeFilterType(int idx);
	void changeKind();

private:
	ComputeReflectivityWidget* m_oriWidget;
	long m_kindId;
	QComboBox* m_typeComboBox;
	QLineEdit* m_nameLineEdit;

	int m_NAME_INDEX = 0;
	int m_KIND_INDEX = 1;
};

class TfpHeaderCellReflectivity : public QWidget {
	Q_OBJECT
public:
	TfpHeaderCellReflectivity(ComputeReflectivityWidget* oriWidget, QWidget *parent = nullptr,
			Qt::WindowFlags f = Qt::WindowFlags());
	~TfpHeaderCellReflectivity();
private slots:
	void changeName();
private:
	ComputeReflectivityWidget* m_oriWidget;
	QLineEdit* m_nameLineEdit;
};

class WellKindCellReflectivity: public QWidget {
	Q_OBJECT
public:
	WellKindCellReflectivity(ComputeReflectivityWidget* oriWidget, long wellId, long kindId,
			QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~WellKindCellReflectivity();

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
	ComputeReflectivityWidget* m_oriWidget;

	QLabel* m_nameLabel;
	QPushButton* m_menuButton;
};

class WellTfpCellReflectivity: public QWidget {
	Q_OBJECT
public:
	WellTfpCellReflectivity(ComputeReflectivityWidget* oriWidget, long wellId,
			QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~WellTfpCellReflectivity();

	long wellId() const;

public slots:
	void updateName();

signals:
	void askChangeTfp();

private slots:
	void askChangeTfpSlot();

private:
	long m_wellId;
	ComputeReflectivityWidget* m_oriWidget;

	QLabel* m_nameLabel;
	QPushButton* m_menuButton;
};

class LogSelectorTreeDialogReflectivity : public QDialog {
	Q_OBJECT
public:
	LogSelectorTreeDialogReflectivity(FolderData* wellFolder, QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~LogSelectorTreeDialogReflectivity();

	WellBore* selectedData() const;

private slots:
	void tryAccept();
	void treeSelectionChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous);

private:
	void updateAcceptButtons();

	FolderData* m_dataSource;
	WellBore* m_selectedData;

	QTreeWidget* m_treeWidget;
	QDialogButtonBox* m_buttonBox;
};

class ComputeReflectivityWidget : public QWidget {
	Q_OBJECT
public:
	enum FilterType {
		Name = 1,
		Kind = 2
	};

	struct WellData {
		// well
		QString wellHeadName;
		QString wellHeadPath; // well head desc path
		QString wellBoreName;
		QString wellBorePath; // well bore desc path
		QString tfpName;
		QString tfpPath;

		// attribute log
		QString attributeName;
		//QString attributeKind;
		QString attributePath;

		// velocity log
		QString velocityName;
		//QString velocityKind;
		QString velocityPath;
	};

	struct HeaderData {
		QString searchName;
		FilterType type = FilterType::Kind;
	};

	// if wellFolder is destroyed, this widget will also be destroyed
	ComputeReflectivityWidget(FolderData* wellFolder, QWidget* parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~ComputeReflectivityWidget();

	void addMoreWells();
	std::vector<long> detectEmptyWells();
	void fillIncompleteWells();

	const std::map<long, WellData>& selection() const;
	const std::array<HeaderData, 2>& header() const;
	QString tfpName() const;
	void setTfpName(const QString& name);

	long attributeId() const;
	long velocityId() const;

	bool changeKind(long id, const HeaderData& header);

	void compute();

	// TODO add signal at the end of the computation

private:
	void frequencyChanged(double value);
	void sampleRateChanged(double value);
	void nameChanged();
	void kindChanged();
	void useRickerChanged(int state);
	QString getDefaultName();
	void triggerDelete();
	void initLogsLayout();
	long addNewWell();
	void addKinds();
	WellHeaderCellReflectivity* createWellCell(long wellId);
	KindHeaderCellReflectivity* createKindCell(long kindId);
	WellKindCellReflectivity* createWellKindCell(long wellId, long kindId);
	void renameWellKindCell(WellKindCellReflectivity* cell);
	void renameWellTfpCell(WellTfpCellReflectivity* cell);
	void deleteWellHeaderCell(WellHeaderCellReflectivity* headerCell, int lineIndex);
	void changeWellWellHeaderCell(WellHeaderCellReflectivity* headerCell);
	void changeWellInData(long wellId, WellBore* wellBore);
	void initLogsMap(WellData& well, WellBore* wellBore);
	void initTfpsMap(WellData& well, WellBore* wellBore);

	const std::vector<std::pair<QString, QString>>& getWellTfps(QString wellBoreDescPath);

	QDoubleSpinBox* m_freqSpinBox;
	QDoubleSpinBox* m_sampleRateSpinBox;
	QLineEdit* m_nameLineEdit;
	QLineEdit* m_kindLineEdit;
	QPushButton* m_addWell;
	QGridLayout* m_logsLayout;

	std::map<long, WellHeaderCellReflectivity*> m_wellHeaderCells;
	std::map<long, std::map<long, WellKindCellReflectivity*>> m_wellKindCells;
	std::map<long, WellTfpCellReflectivity*> m_wellTfpCells;

	// because tfp selection in wellbore is not enough, it needs to be rescanned from directories
	// this cache allow to store the result of the scan
	// key is the wellBore desc path
	// pair first value is the absolute path, second value is the name
	std::map<QString, std::vector<std::pair<QString, QString>>> m_cacheWellTfpCache;

	QString m_outputName;
	QString m_outputKind;
	double m_freq = 75;
	double m_sampleRate = 0.1;
	bool m_useRicker = true;

	std::map<long, WellData> m_selection;
	FolderData* m_dataProvider;

	std::array<HeaderData, 2> m_header;
	QString m_tfpName;

	long m_attributeId = 0;
	long m_velocityId = 1;

	long m_nextId = 0;

	QString m_defaultPrefix = "synth_";
};

#endif // SRC_WIDGET_COMPUTEREFLECTIVITYWIDGET_H

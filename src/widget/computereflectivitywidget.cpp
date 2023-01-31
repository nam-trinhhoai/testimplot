#include "computereflectivitywidget.h"
#include "folderdata.h"
#include "ProjectManagerNames.h"
#include "stringselectordialog.h"
#include "wellbore.h"
#include "wellhead.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>

WellHeaderCellReflectivity::WellHeaderCellReflectivity(ComputeReflectivityWidget* oriWidget, long wellId,
			QWidget *parent, Qt::WindowFlags f) :
				QWidget(parent, f) {
	m_oriWidget = oriWidget;
	m_wellId = wellId;

	QHBoxLayout* mainLayout = new QHBoxLayout;
	setLayout(mainLayout);
	m_wellLabel = new QLabel();
	mainLayout->addWidget(m_wellLabel);
	m_menuButton = new QPushButton("...");
	mainLayout->addWidget(m_menuButton);

	updateName();

	connect(m_menuButton, &QPushButton::clicked, this, &WellHeaderCellReflectivity::openMenu);
}

WellHeaderCellReflectivity::~WellHeaderCellReflectivity() {

}

long WellHeaderCellReflectivity::wellId() const {
	return m_wellId;
}

void WellHeaderCellReflectivity::updateName() {
	const ComputeReflectivityWidget::WellData& wellData = m_oriWidget->selection().at(m_wellId);
	m_wellLabel->setText("[" + wellData.wellHeadName + "] " + wellData.wellBoreName);
}

void WellHeaderCellReflectivity::openMenu() {
	QMenu menu;

	menu.addAction("Change well", this, &WellHeaderCellReflectivity::askChangeWellSlot);
	menu.addAction("Remove", this, &WellHeaderCellReflectivity::askDeleteSlot);

	menu.exec(QCursor::pos());
}

void WellHeaderCellReflectivity::askChangeWellSlot() {
	emit askChangeWell();
}

void WellHeaderCellReflectivity::askDeleteSlot() {
	emit askDelete();
}

KindHeaderCellReflectivity::KindHeaderCellReflectivity(ComputeReflectivityWidget* oriWidget, long kindId, QWidget *parent,
			Qt::WindowFlags f) : QWidget(parent) {
	m_oriWidget = oriWidget;
	m_kindId = kindId;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QHBoxLayout* nameLayout = new QHBoxLayout;
	mainLayout->addLayout(nameLayout);

	m_typeComboBox = new QComboBox;
	m_typeComboBox->addItem("Name");
	m_typeComboBox->addItem("Kind");
	if (m_oriWidget->header().at(m_kindId).type==ComputeReflectivityWidget::FilterType::Name) {
		m_typeComboBox->setCurrentIndex(m_NAME_INDEX);
	} else {
		m_typeComboBox->setCurrentIndex(m_KIND_INDEX);
	}
	nameLayout->addWidget(m_typeComboBox);
	m_nameLineEdit = new QLineEdit(m_oriWidget->header().at(m_kindId).searchName);
	nameLayout->addWidget(m_nameLineEdit);

	connect(m_typeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &KindHeaderCellReflectivity::changeFilterType);
	connect(m_nameLineEdit, &QLineEdit::editingFinished, this, &KindHeaderCellReflectivity::changeKind);
}

KindHeaderCellReflectivity::~KindHeaderCellReflectivity() {

}

long KindHeaderCellReflectivity::kindId() const {
	return m_kindId;
}

void KindHeaderCellReflectivity::changeFilterType(int idx) {
	ComputeReflectivityWidget::HeaderData header = m_oriWidget->header().at(m_kindId);
	if (m_NAME_INDEX==idx) {
		header.type = ComputeReflectivityWidget::FilterType::Name;
	} else {
		header.type = ComputeReflectivityWidget::FilterType::Kind;
	}
	m_oriWidget->changeKind(m_kindId, header);
}

void KindHeaderCellReflectivity::changeKind() {
	ComputeReflectivityWidget::HeaderData header = m_oriWidget->header().at(m_kindId);
	header.searchName = m_nameLineEdit->text();
	m_oriWidget->changeKind(m_kindId, header);
}

TfpHeaderCellReflectivity::TfpHeaderCellReflectivity(ComputeReflectivityWidget* oriWidget, QWidget *parent,
			Qt::WindowFlags f) : QWidget(parent) {
	m_oriWidget = oriWidget;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_nameLineEdit = new QLineEdit(m_oriWidget->tfpName());
	mainLayout->addWidget(m_nameLineEdit);

	connect(m_nameLineEdit, &QLineEdit::editingFinished, this, &TfpHeaderCellReflectivity::changeName);
}

TfpHeaderCellReflectivity::~TfpHeaderCellReflectivity() {

}

void TfpHeaderCellReflectivity::changeName() {
	QString tfpName = m_nameLineEdit->text();
	m_oriWidget->setTfpName(tfpName);
}

WellKindCellReflectivity::WellKindCellReflectivity(ComputeReflectivityWidget* oriWidget, long wellId, long kindId,
			QWidget *parent, Qt::WindowFlags f) : QWidget(parent, f) {
	m_oriWidget = oriWidget;
	m_wellId = wellId;
	m_kindId = kindId;

	QHBoxLayout* mainLayout = new QHBoxLayout;
	setLayout(mainLayout);

	m_nameLabel = new QLabel();
	mainLayout->addWidget(m_nameLabel);

	m_menuButton = new QPushButton("...");
	mainLayout->addWidget(m_menuButton);

	updateName();

	connect(m_menuButton, &QPushButton::clicked, this, &WellKindCellReflectivity::openMenu);
}

WellKindCellReflectivity::~WellKindCellReflectivity() {

}

long WellKindCellReflectivity::wellId() const {
	return m_wellId;
}

long WellKindCellReflectivity::kindId() const {
	return m_kindId;
}

void WellKindCellReflectivity::updateName() {
	QString logName;
	const ComputeReflectivityWidget::WellData& wellData = m_oriWidget->selection().at(m_wellId);
	if (m_kindId==m_oriWidget->attributeId()) {
		logName = wellData.attributeName;
	} else {
		logName = wellData.velocityName;
	}
	m_nameLabel->setText(logName);
}

void WellKindCellReflectivity::openMenu() {
	QMenu menu;

	menu.addAction("Change log", this, &WellKindCellReflectivity::askChangeLogSlot);

	menu.exec(QCursor::pos());
}

void WellKindCellReflectivity::askChangeLogSlot() {
	emit askChangeLog();
}

WellTfpCellReflectivity::WellTfpCellReflectivity(ComputeReflectivityWidget* oriWidget, long wellId,
			QWidget *parent, Qt::WindowFlags f) : QWidget(parent, f) {
	m_oriWidget = oriWidget;
	m_wellId = wellId;

	QHBoxLayout* mainLayout = new QHBoxLayout;
	setLayout(mainLayout);

	m_nameLabel = new QLabel();
	mainLayout->addWidget(m_nameLabel);

	m_menuButton = new QPushButton("...");
	mainLayout->addWidget(m_menuButton);

	updateName();

	connect(m_menuButton, &QPushButton::clicked, this, &WellTfpCellReflectivity::askChangeTfpSlot);
}

WellTfpCellReflectivity::~WellTfpCellReflectivity() {

}

long WellTfpCellReflectivity::wellId() const {
	return m_wellId;
}

void WellTfpCellReflectivity::updateName() {
	QString tfpName;
	const ComputeReflectivityWidget::WellData& wellData = m_oriWidget->selection().at(m_wellId);
	tfpName = wellData.tfpName;
	m_nameLabel->setText(tfpName);
}

void WellTfpCellReflectivity::askChangeTfpSlot() {
	emit askChangeTfp();
}

LogSelectorTreeDialogReflectivity::LogSelectorTreeDialogReflectivity(FolderData* wellFolder,
		QWidget *parent, Qt::WindowFlags f) : QDialog(parent, f) {
	m_dataSource = wellFolder;
	m_selectedData = nullptr;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_treeWidget = new QTreeWidget;
	m_treeWidget->setColumnCount(1);
	mainLayout->addWidget(m_treeWidget);

	QList<QTreeWidgetItem*> wellHeadItems;
	QList<IData*> wellHeads = m_dataSource->data();
	for (int headIndex=0; headIndex<wellHeads.size(); headIndex++) {
		WellHead* wellHead = dynamic_cast<WellHead*>(wellHeads[headIndex]);
		if (wellHead==nullptr) {
			continue;
		}

		QStringList strings;
		strings << wellHead->name();
		QTreeWidgetItem* headItem = new QTreeWidgetItem(static_cast<QTreeWidgetItem*>(nullptr), strings);
		headItem->setFlags(headItem->flags() & ~Qt::ItemIsSelectable);
		headItem->setData(0, Qt::UserRole, headIndex);

		for (int boreIndex=0; boreIndex<wellHead->wellBores().size(); boreIndex++) {
			WellBore* wellBore = wellHead->wellBores()[boreIndex];

			QStringList boreStrings;
			boreStrings << wellBore->name();
			QTreeWidgetItem* boreItem = new QTreeWidgetItem(headItem, boreStrings);
			boreItem->setData(0, Qt::UserRole, boreIndex);
		}

		wellHeadItems.push_back(headItem);
	}
	m_treeWidget->insertTopLevelItems(0, wellHeadItems);

	m_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok |
			QDialogButtonBox::Cancel);
	mainLayout->addWidget(m_buttonBox);

	connect(m_treeWidget, &QTreeWidget::currentItemChanged, this, &LogSelectorTreeDialogReflectivity::treeSelectionChanged);

	connect(m_buttonBox, &QDialogButtonBox::accepted, this, &LogSelectorTreeDialogReflectivity::tryAccept);
	connect(m_buttonBox, &QDialogButtonBox::rejected, this, &LogSelectorTreeDialogReflectivity::reject);

	updateAcceptButtons();
}

LogSelectorTreeDialogReflectivity::~LogSelectorTreeDialogReflectivity() {

}

WellBore* LogSelectorTreeDialogReflectivity::selectedData() const {
	return m_selectedData;
}

void LogSelectorTreeDialogReflectivity::tryAccept() {
	if (m_selectedData!=nullptr) {// use indexes to see if a data is selected
		accept();
	}
}

void LogSelectorTreeDialogReflectivity::updateAcceptButtons() {
	if (m_selectedData!=nullptr) {// use indexes to see if a data is selected
		m_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
	} else {
		m_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
	}
}

void LogSelectorTreeDialogReflectivity::treeSelectionChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous) {
	if (current==nullptr || current->childCount()!=0) {
		m_selectedData = nullptr;
	} else {
		bool headValid, boreValid;
		long wellHeadIdx = current->parent()->data(0, Qt::UserRole).toInt(&headValid);
		long wellBoreIdx = current->data(0, Qt::UserRole).toInt(&boreValid);

		QList<IData*> wellHeads = m_dataSource->data();
		bool valid = headValid && boreValid && wellHeadIdx>=00 && wellHeadIdx<wellHeads.size() && wellBoreIdx>=0;

		WellHead* wellHead = nullptr;
		if (valid) {
			wellHead = dynamic_cast<WellHead*>(wellHeads[wellHeadIdx]);
			valid = wellHead!=nullptr && wellBoreIdx<wellHead->wellBores().size();
		}

		if (valid) {
			m_selectedData = wellHead->wellBores()[wellBoreIdx];
		} else {
			m_selectedData = nullptr;
		}
	}

	updateAcceptButtons();
}

ComputeReflectivityWidget::ComputeReflectivityWidget(FolderData* wellFolder, QWidget* parent, Qt::WindowFlags f) :
			QWidget(parent, f) {
	m_dataProvider = wellFolder;
	setAttribute(Qt::WA_DeleteOnClose);

	m_outputName = getDefaultName();
	m_outputKind = m_outputName;

	m_header[m_attributeId].searchName = "RHOB";
	m_header[m_velocityId].searchName = "DT";
	m_tfpName = "tfp";

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_logsLayout = new QGridLayout;
	QWidget* logsHolder = new QWidget;
	logsHolder->setLayout(m_logsLayout);
	QScrollArea* logsScrollArea = new QScrollArea;
	logsScrollArea->setWidget(logsHolder);
	logsScrollArea->setWidgetResizable(true);
	mainLayout->addWidget(logsScrollArea);

	QPushButton* addMoreWells = new QPushButton("Add more wells");
	mainLayout->addWidget(addMoreWells);

	QFormLayout* formLayout = new QFormLayout;
	mainLayout->addLayout(formLayout);

	m_nameLineEdit = new QLineEdit(m_outputName);
	formLayout->addRow("Output Log Name", m_nameLineEdit);

	m_kindLineEdit = new QLineEdit(m_outputKind);
	formLayout->addRow("Output Log Kind", m_kindLineEdit);

	m_freqSpinBox = new QDoubleSpinBox;
	m_freqSpinBox->setMinimum(std::numeric_limits<float>::min());
	m_freqSpinBox->setMaximum(std::numeric_limits<float>::max());
	m_freqSpinBox->setValue(m_freq);
	formLayout->addRow("Frequency", m_freqSpinBox);

	m_sampleRateSpinBox = new QDoubleSpinBox;
	m_sampleRateSpinBox->setMinimum(std::numeric_limits<float>::min());
	m_sampleRateSpinBox->setMaximum(std::numeric_limits<float>::max());
	m_sampleRateSpinBox->setValue(m_sampleRate);
	formLayout->addRow("Sampling rate", m_sampleRateSpinBox);

	QCheckBox* computeMethodCheckBox = new QCheckBox();
	computeMethodCheckBox->setCheckState(m_useRicker ? Qt::Checked : Qt::Unchecked);
	formLayout->addRow("Use Ricker", computeMethodCheckBox);


	QPushButton* computeButton = new QPushButton("Compute");
	mainLayout->addWidget(computeButton);

	connect(m_nameLineEdit, &QLineEdit::editingFinished, this, &ComputeReflectivityWidget::nameChanged);
	connect(m_kindLineEdit, &QLineEdit::editingFinished, this, &ComputeReflectivityWidget::kindChanged);
	connect(m_freqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ComputeReflectivityWidget::frequencyChanged);
	connect(m_sampleRateSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ComputeReflectivityWidget::sampleRateChanged);
	connect(computeMethodCheckBox, &QCheckBox::stateChanged, this, &ComputeReflectivityWidget::useRickerChanged);
	connect(m_dataProvider, &QObject::destroyed, this, &ComputeReflectivityWidget::triggerDelete);
	connect(computeButton, &QPushButton::clicked, this, &ComputeReflectivityWidget::compute);
	connect(addMoreWells, &QPushButton::clicked, this, &ComputeReflectivityWidget::addMoreWells);
	initLogsLayout();
}

ComputeReflectivityWidget::~ComputeReflectivityWidget() {

}

std::vector<long> ComputeReflectivityWidget::detectEmptyWells() {
	std::vector<long> emptyWellIds;
	std::map<long, WellData>::const_iterator it = m_selection.begin();
	while (it!=m_selection.end()) {
		if (it->second.wellBorePath.isNull() || it->second.wellBorePath.isEmpty()) {
			emptyWellIds.push_back(it->first);
		}
		it++;
	}
	return emptyWellIds;
}

void ComputeReflectivityWidget::fillIncompleteWells() {
	// TODO
}

void ComputeReflectivityWidget::addMoreWells() {
	// Can be optimized by first extracting a list of WellParameter
	// and if we assume that there is no duplicate in WELLMASTER
	// That will avoid checking newly added wells

	fillIncompleteWells();

	// Detect empty lines
	std::vector<long> emptyWellIds = detectEmptyWells();

	QList<IData*> wellHeads = m_dataProvider->data();
	for (int idxHead=0; idxHead<wellHeads.size(); idxHead++) {
		WellHead* wellHead = dynamic_cast<WellHead*>(wellHeads[idxHead]);
		if (wellHead==nullptr) {
			continue;
		}

		QList<WellBore*> wellBores = wellHead->wellBores();
		for (int idxBore=0; idxBore<wellBores.size(); idxBore++) {
			WellBore* wellBore = wellBores[idxBore];

			// check if already added
			bool isNotAdded = true;
			std::map<long, WellData>::const_iterator itAdded = m_selection.begin();
			while (isNotAdded && itAdded!=m_selection.end()) {
				isNotAdded = itAdded->second.wellBorePath.compare(wellBore->getDescPath())!=0;
				itAdded++;
			}

			const std::vector<QString>& logFiles = wellBore->logsFiles();
			const std::vector<QString>& logNames = wellBore->logsNames();

			// get logs
			bool isLogValid = true;
			std::map<long, int> idToIndex;
			if (isNotAdded) {
				int logHeaderIdx = 0;
				while (isLogValid && logHeaderIdx<m_header.size()) {
					QString itFilterName = m_header[logHeaderIdx].searchName.toLower();
					isLogValid = false;
					int iLog = 0;
					while (!isLogValid && iLog<logFiles.size()) {//iLog<wellBoreList.log_tinyname.size()) {
						if (m_header[logHeaderIdx].type==FilterType::Name) {
							isLogValid = itFilterName.compare(logNames[iLog].toLower())==0;
						} else {
							QString kind = WellBore::getKindFromLogFile(logFiles[iLog]);
							isLogValid = itFilterName.compare(kind.toLower())==0;
						}
						if (isLogValid) {
							idToIndex[logHeaderIdx] = iLog;
						}
						iLog++;
					}
					logHeaderIdx++;
				}
			}

			QString tfpPath = "";
			QString tfpName = "";
			bool foundTfp = false;
			if (isNotAdded && isLogValid && !m_tfpName.isNull() && !m_tfpName.isEmpty()) {
				// search the one matching the tfp name
				bool tfpNotFound = true;
				const std::vector<std::pair<QString, QString>>& tfps = getWellTfps(wellBore->getDescPath());

				int tfpIndex=0;
				while (tfpNotFound && tfpIndex<tfps.size()) {
					tfpNotFound = m_tfpName.compare(tfps[tfpIndex].second)!=0;
					if (tfpNotFound) {
						tfpIndex++;
					}
				}
				if (!tfpNotFound) {
					tfpName = tfps[tfpIndex].second;
					tfpPath = tfps[tfpIndex].first;
					foundTfp = true;
				} else {
					// if tfp filter is valid but the tfp was not found, do not add the well
					isLogValid = false;
				}
			}

			// check log
			if (isNotAdded && isLogValid) {
				// add
				long id;
				if (emptyWellIds.size()>0) {
					id = emptyWellIds[0];
					emptyWellIds.erase(emptyWellIds.begin());
				} else {
					id = addNewWell();
				}
				changeWellInData(id, wellBore);
				WellData wellParam = m_selection.at(id);
				for (std::pair<long, int> idAndIndex : idToIndex) {
					if (idAndIndex.first==m_attributeId) {
						wellParam.attributePath = logFiles[idAndIndex.second];
						wellParam.attributeName = logNames[idAndIndex.second];
					} else if (idAndIndex.first==m_velocityId) {
						wellParam.velocityPath = logFiles[idAndIndex.second];
						wellParam.velocityName = logNames[idAndIndex.second];
					}
				}
				if (foundTfp) {
					wellParam.tfpPath = tfpPath;
					wellParam.tfpName = tfpName;
				}
				m_selection[id] = wellParam;
				m_wellHeaderCells[id]->updateName();
				std::map<long, WellKindCellReflectivity*>::iterator it = m_wellKindCells[id].begin();
				while (it!=m_wellKindCells[id].end()) {
					it->second->updateName();
					it++;
				}
				m_wellTfpCells[id]->updateName();
			}
		}
	}
}

const std::map<long, ComputeReflectivityWidget::WellData>& ComputeReflectivityWidget::selection() const {
	return m_selection;
}

const std::array<ComputeReflectivityWidget::HeaderData, 2>& ComputeReflectivityWidget::header() const {
	return m_header;
}

QString ComputeReflectivityWidget::tfpName() const {
	return m_tfpName;
}

void ComputeReflectivityWidget::setTfpName(const QString& name) {
	m_tfpName = name;
}

long ComputeReflectivityWidget::attributeId() const {
	return m_attributeId;
}

long ComputeReflectivityWidget::velocityId() const {
	return m_velocityId;
}

bool ComputeReflectivityWidget::changeKind(long id, const HeaderData& header) {
	bool out = true;
	if (id==m_attributeId || id==m_velocityId) {
		m_header[id] = header;
	} else {
		out = false;
	}
	return out;
}

void ComputeReflectivityWidget::compute() {
	bool noToAll = false;
	bool okToAll = false;

	QStringList errorMessages;
	std::map<long, WellData>::const_iterator it = m_selection.cbegin();
	for (std::map<long, WellData>::const_iterator it = m_selection.cbegin();
			it!=m_selection.cend(); it++) {
		WellHead* wellHead = WellHead::getWellHeadFromDescFile(it->second.wellHeadPath, nullptr, this);

		if (wellHead==nullptr) {
			continue;
		}

		QString deviationFile = QFileInfo(it->second.wellBorePath).dir().absoluteFilePath("deviation");
		std::vector<QString> tfpName, tfpPath, logName, logPath;
		tfpName.push_back(it->second.tfpName);
		tfpPath.push_back(it->second.tfpPath);
		WellBore* wellBore = new WellBore(nullptr, it->second.wellBorePath, deviationFile,
				tfpPath, tfpName, logPath, logName, wellHead, this);

		QString newLogPath = QFileInfo(it->second.wellBorePath).dir().absoluteFilePath(m_outputName + ".log");
		bool doCompute = !QFileInfo(it->second.wellBorePath).exists(newLogPath);
		if (!doCompute) {
			if (okToAll) {
				doCompute = true;
			} else if (noToAll) {
				doCompute = false;
			} else {
				QString okItem = "Overwrite";
				QString okToAllItem = "Overwrite all wells";
				QString noItem = "Skip this well";
				QString noToAllItem = "Skip all wells";
				QStringList items;
				items << noItem << noToAllItem << okItem << okToAllItem;
				QString res = QInputDialog::getItem(this, "Overwrite ?", "Log file "+m_outputName+" for well ["+it->second.wellHeadName+"] "+
						it->second.wellBoreName+" alreay exists, should it be overwritten ?", items);
				okToAll = res.compare(okToAllItem)==0;
				noToAll = res.compare(noToAllItem)==0;
				doCompute = res.compare(okItem)==0 || okToAll;
			}
		}
		if (doCompute) {
			ReflectivityError err = wellBore->computeReflectivity(it->second.attributePath, it->second.velocityPath, m_sampleRate,
					m_freq, m_useRicker, m_outputName, m_outputKind, newLogPath);

			if (err!=ReflectivityError::NoError) {
				QString errStr;
				if (err==ReflectivityError::NoTfp) {
					errStr = "Provided tfp is invalid : " + it->second.tfpName;
				} else if (err==ReflectivityError::TfpNotReversible) {
					errStr = "Provided tfp is not reversible  : " + it->second.tfpName;
				} else if (err==ReflectivityError::AttributeLogNotValid) {
					errStr = "Provided "+m_header[m_attributeId].searchName+" is invalid : " + it->second.attributeName;
				} else if (err==ReflectivityError::VelocityLogNotValid) {
					errStr = "Provided "+m_header[m_velocityId].searchName+" is invalid : " + it->second.velocityName;
				} else if (err==ReflectivityError::NoLogIntervalIntersection) {
					errStr = "Provided input logs is invalid do not intersect : " + it->second.attributeName + " " + it->second.velocityName;
				} else if (err==ReflectivityError::OnlyInvalidIntervals) {
					errStr = "Provided input logs is invalid there is an intersection but an error appeared during twt to log conversion : " + it->second.attributeName + " " + it->second.velocityName;
				} else if (err==ReflectivityError::FailToWriteLog) {
					errStr = "Failed to write output reflectivity log : " + m_outputName;
				}
				errStr = "[" + it->second.wellHeadName + "] " + it->second.wellBoreName + " " + errStr;
				errorMessages << errStr;
			}
		}
		wellBore->deleteLater();
		wellHead->deleteLater();
	}

	if (errorMessages.size()>0) {
		QMessageBox::warning(this, "Reflectivity computation : Error occured", errorMessages.join("\n"));
	} else {
		QMessageBox::information(this, "Reflectivity computation : Finished", "Computation finished");
	}
}

QString ComputeReflectivityWidget::getDefaultName() {
	if (m_useRicker) {
		return m_defaultPrefix + QString::number(m_freq) + "_rick";
	} else {
		return m_defaultPrefix + QString::number(m_freq) + "_bpf";
	}
}

void ComputeReflectivityWidget::nameChanged() {
	m_outputName = m_nameLineEdit->text();
}

void ComputeReflectivityWidget::kindChanged() {
	m_outputKind = m_kindLineEdit->text();
}

void ComputeReflectivityWidget::frequencyChanged(double value) {
	QString defaultName = getDefaultName();
	bool redoName = defaultName.compare(m_outputName)==0;
	bool redoKind = defaultName.compare(m_outputKind)==0;
	m_freq = value;

	QString newDefaultName = getDefaultName();
	if (redoName) {
		m_outputName = newDefaultName;
		m_nameLineEdit->setText(m_outputName);
	}
	if (redoKind) {
		m_outputKind = newDefaultName;
		m_kindLineEdit->setText(m_outputKind);
	}
}

void ComputeReflectivityWidget::sampleRateChanged(double value) {
	m_sampleRate = value;
}

void ComputeReflectivityWidget::useRickerChanged(int state) {
	QString defaultName = getDefaultName();
	bool redoName = defaultName.compare(m_outputName)==0;
	bool redoKind = defaultName.compare(m_outputKind)==0;
	m_useRicker = state==Qt::Checked;

	QString newDefaultName = getDefaultName();
	if (redoName) {
		m_outputName = newDefaultName;
		m_nameLineEdit->setText(m_outputName);
	}
	if (redoKind) {
		m_outputKind = newDefaultName;
		m_kindLineEdit->setText(m_outputKind);
	}
}

void ComputeReflectivityWidget::triggerDelete() {
	this->deleteLater();
}

void ComputeReflectivityWidget::initLogsLayout() {
	m_addWell = new QPushButton(QIcon(":/slicer/icons/add.png"), "");
	m_addWell->setToolTip("Add new well");
	m_logsLayout->addWidget(m_addWell, 0, 1);

	connect(m_addWell, &QPushButton::clicked, this, &ComputeReflectivityWidget::addNewWell);

	addKinds(); // add kind does not create kind well cells
	addNewWell();
}

long ComputeReflectivityWidget::addNewWell() {
	long wellId = m_nextId++;
	WellData newWellData;
	m_selection[wellId] = newWellData;
	long lineIndex = wellId+1;
	WellHeaderCellReflectivity* headerCell = createWellCell(wellId);
	m_logsLayout->removeWidget(m_addWell);
	m_logsLayout->addWidget(headerCell, lineIndex, 0);
	m_logsLayout->addWidget(m_addWell, lineIndex+1, 0);

	m_wellHeaderCells[wellId] = headerCell;

	// work because ids are increasing, else changes of order may appear
	for (int i=0; i<2; i++) {
		WellKindCellReflectivity* cell = createWellKindCell(wellId, i);
		m_logsLayout->addWidget(cell, lineIndex, i+2);

		m_wellKindCells[wellId][i] = cell;

		connect(cell, &WellKindCellReflectivity::askChangeLog, [this, cell]() {
			renameWellKindCell(cell);
		});
	}

	WellTfpCellReflectivity* tfpCell = new WellTfpCellReflectivity(this, wellId);
	m_logsLayout->addWidget(tfpCell, lineIndex, 1);
	m_wellTfpCells[wellId] = tfpCell;
	connect(tfpCell, &WellTfpCellReflectivity::askChangeTfp, [this, tfpCell]() {
		renameWellTfpCell(tfpCell);
	});

	connect(headerCell, &WellHeaderCellReflectivity::askDelete, [this, headerCell, lineIndex]() {
		deleteWellHeaderCell(headerCell, lineIndex);
	});
	connect(headerCell, &WellHeaderCellReflectivity::askChangeWell, [this, headerCell]() {
		changeWellWellHeaderCell(headerCell);
	});

	return wellId;
}

void ComputeReflectivityWidget::addKinds() {
	KindHeaderCellReflectivity* attributeHeaderCell = createKindCell(0);
	KindHeaderCellReflectivity* velocityHeaderCell = createKindCell(1);
	TfpHeaderCellReflectivity* tfpHeaderCell = new TfpHeaderCellReflectivity(this);
	m_logsLayout->addWidget(attributeHeaderCell, 0, 2);
	m_logsLayout->addWidget(velocityHeaderCell, 0, 3);
	m_logsLayout->addWidget(tfpHeaderCell, 0, 1);
}

WellHeaderCellReflectivity* ComputeReflectivityWidget::createWellCell(long wellId) {
	return new WellHeaderCellReflectivity(this, wellId);
}

KindHeaderCellReflectivity* ComputeReflectivityWidget::createKindCell(long kindId) {
	return new KindHeaderCellReflectivity(this, kindId);
}

WellKindCellReflectivity* ComputeReflectivityWidget::createWellKindCell(long wellId, long kindId) {
	return new WellKindCellReflectivity(this, wellId, kindId);
}

void ComputeReflectivityWidget::renameWellKindCell(WellKindCellReflectivity* cell) {
	bool badParam = m_selection.find(cell->wellId())==m_selection.end();
	if (!badParam) {
		const WellData& wellParameter = m_selection.at(cell->wellId());
		badParam = cell->kindId()!=m_attributeId && cell->kindId()!=m_velocityId;
	}
	if (badParam) {
		qDebug() << "renameWellKindCell : Cell does not match data";
		return;
	}

	QList<IData*> wellHeads = m_dataProvider->data();
	int wellHeadIdx=0, wellBoreIdx=0;
	const WellData& wellParameter = m_selection.at(cell->wellId());

	bool valid = false;
	WellBore* wellBore = nullptr;
	while (!valid && wellHeadIdx<wellHeads.size()) {
		WellHead* wellHead = dynamic_cast<WellHead*>(wellHeads[wellHeadIdx]);
		if (wellHead==nullptr) {
			continue;
		}
		QList<WellBore*> wellBores = wellHead->wellBores();

		wellBoreIdx=0;
		while (!valid && wellBoreIdx<wellBores.size()) {
			valid = wellParameter.wellBorePath.compare(wellBores[wellBoreIdx]->getDescPath())==0;
			if (!valid) {
				wellBoreIdx++;
			} else {
				wellBore = wellBores[wellBoreIdx];
			}
		}

		if (!valid) {
			wellHeadIdx++;
		}
	}

	if (valid) {
		std::vector<QString> logNames = wellBore->logsNames();
		std::vector<QString> logFiles = wellBore->logsFiles();
		QStringList logNamesList(logNames.begin(), logNames.end());// = wellBoreList.log_tinyname;
		StringSelectorDialog dialog(&logNamesList, "Select log");
		int code = dialog.exec();
		bool accepted = code==QDialog::Accepted;
		if (accepted) {
			int newLogIndex = dialog.getSelectedIndex();

			if (newLogIndex>=0 && newLogIndex<logNames.size()) {
				WellData newWellParameter = wellParameter;
				if (cell->kindId()==m_attributeId) {
					newWellParameter.attributeName = logNames[newLogIndex];
					newWellParameter.attributePath = logFiles[newLogIndex];
				} else if (cell->kindId()==m_velocityId) {
					newWellParameter.velocityName = logNames[newLogIndex];
					newWellParameter.velocityPath = logFiles[newLogIndex];
				}
				m_selection[cell->wellId()] = newWellParameter;
				cell->updateName();
			}
		}
	}
	if (!valid) {
		qDebug() << "renameWellKindCell : Invalid cell, well basket and data relation";
	}
}

void ComputeReflectivityWidget::renameWellTfpCell(WellTfpCellReflectivity* cell) {
	bool badParam = m_selection.find(cell->wellId())==m_selection.end();
	if (badParam) {
		qDebug() << "renameWellTfpCell : Cell does not match data";
		return;
	}

	QList<IData*> wellHeads = m_dataProvider->data();
	int wellHeadIdx=0, wellBoreIdx=0;
	const WellData& wellParameter = m_selection.at(cell->wellId());

	bool valid = false;
	WellBore* wellBore = nullptr;
	while (!valid && wellHeadIdx<wellHeads.size()) {
		WellHead* wellHead = dynamic_cast<WellHead*>(wellHeads[wellHeadIdx]);
		if (wellHead==nullptr) {
			continue;
		}
		QList<WellBore*> wellBores = wellHead->wellBores();

		wellBoreIdx=0;
		while (!valid && wellBoreIdx<wellBores.size()) {
			valid = wellParameter.wellBorePath.compare(wellBores[wellBoreIdx]->getDescPath())==0;
			if (!valid) {
				wellBoreIdx++;
			} else {
				wellBore = wellBores[wellBoreIdx];
			}
		}

		if (!valid) {
			wellHeadIdx++;
		}
	}

	if (valid) {
		const std::vector<std::pair<QString, QString>>& wellTfps = getWellTfps(wellBore->getDescPath());

		QStringList logNamesList;
		logNamesList.reserve(wellTfps.size());
		for (const std::pair<QString, QString>& pair : wellTfps) {
			logNamesList.append(pair.second);
		}
		StringSelectorDialog dialog(&logNamesList, "Select log");
		int code = dialog.exec();
		bool accepted = code==QDialog::Accepted;
		if (accepted) {
			int newLogIndex = dialog.getSelectedIndex();

			if (newLogIndex>=0 && newLogIndex<wellTfps.size()) {
				WellData newWellParameter = wellParameter;
				newWellParameter.tfpName = wellTfps[newLogIndex].second;
				newWellParameter.tfpPath = wellTfps[newLogIndex].first;
				m_selection[cell->wellId()] = newWellParameter;
				cell->updateName();
			}
		}
	}
	if (!valid) {
		qDebug() << "renameWellTfpCell : Invalid cell, well basket and data relation";
	}
}

void ComputeReflectivityWidget::deleteWellHeaderCell(WellHeaderCellReflectivity* headerCell, int lineIndex) {
	long wellId = headerCell->wellId();
	m_wellHeaderCells.erase(wellId);
	m_wellKindCells.erase(wellId);
	m_wellTfpCells.erase(wellId);

	// delete ui then data
	for (int col=0; col<m_logsLayout->columnCount(); col++) {
		QLayoutItem* layoutItem = m_logsLayout->itemAtPosition(lineIndex, col);
		if (layoutItem!=nullptr && layoutItem->widget()!=nullptr) {
			layoutItem->widget()->deleteLater();
		}
	}

	m_selection.erase(wellId);
}

void ComputeReflectivityWidget::changeWellWellHeaderCell(WellHeaderCellReflectivity* headerCell) {
	QString wellName = "";

	LogSelectorTreeDialogReflectivity dialog(m_dataProvider);

	int outCode = dialog.exec();

	bool valid = outCode==QDialog::Accepted;
	WellBore* wellBore = nullptr;
	if (valid) {
		wellBore = dialog.selectedData();
		valid = wellBore!=nullptr;
	}

	if (valid) {
		QString newDescFile = wellBore->getDescPath();
		QString oldDescFile = m_selection.at(headerCell->wellId()).wellBorePath;

		valid = valid && newDescFile.compare(oldDescFile)!=0;

		if (valid) {
			changeWellInData(headerCell->wellId(), wellBore);
			headerCell->updateName();

			std::map<long, WellKindCellReflectivity*>::iterator it = m_wellKindCells[headerCell->wellId()].begin();
			while (it!=m_wellKindCells[headerCell->wellId()].end()) {
				it->second->updateName();
				it++;
			}
			m_wellTfpCells[headerCell->wellId()]->updateName();
		}
	}
}

void ComputeReflectivityWidget::changeWellInData(long wellId, WellBore* wellBore) {
	WellData well;
	well.wellBoreName = wellBore->name();
	well.wellBorePath = wellBore->getDescPath();
	well.wellHeadName = wellBore->wellHead()->name();
	well.wellHeadPath = wellBore->wellHead()->idPath();

//	well.tfpName = wellBore->getTfpName();
//	well.tfpPath = wellBore->getTfpFilePath();
	initTfpsMap(well, wellBore);
	initLogsMap(well, wellBore);
	m_selection[wellId] = well;
}

void ComputeReflectivityWidget::initLogsMap(ComputeReflectivityWidget::WellData& well, WellBore* wellBore) {
	for (int i=0; i<2; i++) {
		const HeaderData& header = m_header[i];
		QString logPath = "";
		QString logName = "";
		bool logNotFound = true;

		if (!header.searchName.isNull() && !header.searchName.isEmpty()) {
			std::vector<QString> logNames = wellBore->logsNames();
			std::vector<QString> logFiles = wellBore->logsFiles();

			int logIndex=0;
			while (logNotFound && logIndex<logFiles.size()) {
				if (header.type==FilterType::Kind) {
					QString kind = WellBore::getKindFromLogFile(logFiles[logIndex]);
					logNotFound = kind.compare(header.searchName)!=0;
				} else {
					QString name = logNames[logIndex];
					logNotFound = name.compare(header.searchName)!=0;
				}
				if (logNotFound) {
					logIndex++;
				}
			}
			if (!logNotFound) {
				logName = logNames[logIndex];
				logPath = logFiles[logIndex];
			}
		}

		// always add logPath even if empty
		if (i==m_attributeId) {
			well.attributeName = logName;
			well.attributePath = logPath;
		} else if (i==m_velocityId) {
			well.velocityName = logName;
			well.velocityPath = logPath;
		}
	}
}

void ComputeReflectivityWidget::initTfpsMap(ComputeReflectivityWidget::WellData& well, WellBore* wellBore) {
	QString tfpPath = "";
	QString tfpName = "";
	bool tfpNotFound = true;

	// search the one matching the tfp name
	if (!m_tfpName.isNull() && !m_tfpName.isEmpty()) {
		const std::vector<std::pair<QString, QString>>& tfps = getWellTfps(well.wellBorePath);

		int tfpIndex=0;
		while (tfpNotFound && tfpIndex<tfps.size()) {
			tfpNotFound = m_tfpName.compare(tfps[tfpIndex].second)!=0;
			if (tfpNotFound) {
				tfpIndex++;
			}
		}
		if (!tfpNotFound) {
			tfpName = tfps[tfpIndex].second;
			tfpPath = tfps[tfpIndex].first;
		}
	}

	// fall back with data selection
	if (tfpPath.isNull() || tfpPath.isEmpty()) {
		tfpName = wellBore->getTfpName();
		tfpPath = wellBore->getTfpFilePath();
	}

	// fall back with default tfp
	if (tfpPath.isNull() || tfpPath.isEmpty()) {
		QString defaultAbsolutePath = WellBore::getTfpFileFromDescFile(well.wellBorePath);
		QString name = ProjectManagerNames::getKeyTabFromFilename(defaultAbsolutePath, "Name");
		if (!name.isEmpty()) {
			tfpPath = defaultAbsolutePath;
			tfpName = name;
		}
	}

	// always add tfpPath even if empty
	well.tfpName = tfpName;
	well.tfpPath = tfpPath;
}

const std::vector<std::pair<QString, QString>>& ComputeReflectivityWidget::getWellTfps(QString wellBoreDescPath) {
	auto it = m_cacheWellTfpCache.find(wellBoreDescPath);
	if (it!=m_cacheWellTfpCache.end()) {
		return it->second;
	}

	QDir wellBoreDir = QFileInfo(wellBoreDescPath).dir();
	QFileInfoList tfps = wellBoreDir.entryInfoList(QStringList() << "*.tfp", QDir::Files | QDir::Readable);

	std::vector<std::pair<QString, QString>> _tfpsData;
	m_cacheWellTfpCache[wellBoreDescPath] = _tfpsData;
	std::vector<std::pair<QString, QString>>& tfpsData = m_cacheWellTfpCache[wellBoreDescPath];
	for (long i=0; i<tfps.size(); i++) {
		QFileInfo tfpFile = tfps[i];

		QString absolutePath = tfpFile.absoluteFilePath();
		QString name = ProjectManagerNames::getKeyTabFromFilename(absolutePath, "Name");
		if (!name.isEmpty()) {
			tfpsData.push_back(std::pair<QString, QString>(absolutePath, name));
		}
	}

	return m_cacheWellTfpCache[wellBoreDescPath];
}

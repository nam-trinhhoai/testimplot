#include "savehorizondialog.h"

#include "nvlineedit.h"

#include <QListWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QString>
#include <QPushButton>
#include <QListWidget>
#include <QListWidget>
#include <QDialogButtonBox>
#include <QCheckBox>
#include <QLabel>
#include <QTabWidget>

SaveHorizonDialog::SaveHorizonDialog(const QStringList& saveNames, QString title, QWidget *parent,
		Qt::WindowFlags f) : QDialog(parent, f), m_saveNames(saveNames) {

	this->setWindowTitle(title);

	QVBoxLayout* pMainLayout = new QVBoxLayout();
	this->setLayout(pMainLayout);

	QTabWidget* tabWidget = new QTabWidget;
	pMainLayout->addWidget(tabWidget);

	m_listWidget = new QListWidget;
	tabWidget->addTab(m_listWidget, "Existing horizons");
	m_listWidget->setDragEnabled(true);
	m_listWidget->setDropIndicatorShown(true);
	m_listWidget->setDragDropMode(QAbstractItemView::InternalMove);

	foreach(QString const& string, m_saveNames) {
		m_listWidget->addItem(string);
	}
	m_listWidget->setMinimumWidth(m_listWidget->sizeHintForColumn(0)+10);
	m_listWidget->adjustSize();

	QWidget* newNameHolder = new QWidget;
	QVBoxLayout* newNameLayout = new QVBoxLayout;
	newNameHolder->setLayout(newNameLayout);

	m_newNameLineEdit = new NvLineEdit();
	newNameLayout->addWidget(m_newNameLineEdit);
	newNameLayout->addStretch(1);
	tabWidget->addTab(newNameHolder, "New Horizon");

	//QWidget* checkBoxHolder = new QWidget;
	QHBoxLayout* checkBoxLayout = new QHBoxLayout;
	//checkBoxHolder->setLayout(checkBoxLayout);
	pMainLayout->addLayout(checkBoxLayout);

	QLabel* checkBoxLabel = new QLabel("Use Interpolation :");
	checkBoxLayout->addWidget(checkBoxLabel);

	m_interpolationCheckBox = new QCheckBox;
	m_interpolationCheckBox->setCheckState(Qt::Checked);
	checkBoxLayout->addWidget(m_interpolationCheckBox);

	//QWidget* checkBoxHolder = new QWidget;
	QHBoxLayout* rgbBoxLayout = new QHBoxLayout;
	//checkBoxHolder->setLayout(checkBoxLayout);
	pMainLayout->addLayout(rgbBoxLayout);

	QLabel* rgbBoxLabel = new QLabel("Use RGB :");
	rgbBoxLayout->addWidget(rgbBoxLabel);

	m_rgbCheckBox = new QCheckBox;
	m_rgbCheckBox->setCheckState(Qt::Checked);
	rgbBoxLayout->addWidget(m_rgbCheckBox);



	connect(m_listWidget, 	&QListWidget::currentItemChanged,
			this, &SaveHorizonDialog::slotSelectItem);

	connect(m_newNameLineEdit, &QLineEdit::editingFinished, this,
			&SaveHorizonDialog::setNewLayerName);

	connect(m_interpolationCheckBox, &QCheckBox::stateChanged, this,
			&SaveHorizonDialog::updateInterpolationToggle);

	connect(m_rgbCheckBox, &QCheckBox::stateChanged, this,
			&SaveHorizonDialog::updateRgbToggle);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	pMainLayout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

	this->adjustSize();
}

SaveHorizonDialog::~SaveHorizonDialog() {}

QString SaveHorizonDialog::getSaveName() {
	return m_outputSaveName;
}

bool SaveHorizonDialog::doInterpolation() {
	return m_doInterpolation;
}

bool SaveHorizonDialog::doRgb() {
	return m_doRgb;
}

bool SaveHorizonDialog::isNameNew() {
	return m_isNameNew;
}

void SaveHorizonDialog::slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous){
  int count = m_listWidget->count();
  for(int index = 0; index < count; index++){
    QListWidgetItem * item = m_listWidget->item(index);
    QFont font = item->font();
    font.setBold(false);
    item->setFont(font);
  }

  m_outputSaveName = current->text();
  QListWidgetItem* pTemp = m_listWidget->currentItem();
  QFont font = pTemp->font();
  font.setBold(true);
  pTemp->setFont(font);

  QSignalBlocker blocker(m_newNameLineEdit);
  m_newNameLineEdit->setText("");

	m_isNameNew = false;
}

void SaveHorizonDialog::updateInterpolationToggle(int state) {
	m_doInterpolation = state==Qt::Checked;
}

void SaveHorizonDialog::updateRgbToggle(int state) {
	m_doRgb = state==Qt::Checked;
}

void SaveHorizonDialog::setNewLayerName() {
	QList<QListWidgetItem*> selectedItems = m_listWidget->selectedItems();
	if (!m_newNameLineEdit->text().isEmpty() ||
			(m_newNameLineEdit->text().isEmpty() && selectedItems.count()==0)) {
		m_outputSaveName = m_newNameLineEdit->text();
	}
	QSignalBlocker blocker(m_listWidget);
	for (QListWidgetItem* item : selectedItems) {
		QFont font = item->font();
		font.setBold(false);
		item->setFont(font);
		item->setSelected(false);
	}
	m_isNameNew = true;
}

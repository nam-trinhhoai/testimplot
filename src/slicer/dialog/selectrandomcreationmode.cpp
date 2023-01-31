#include "selectrandomcreationmode.h"

#include "workingsetmanager.h"
#include "wellbore.h"
#include "wellhead.h"
#include "folderdata.h"

#include <QListWidget>
#include <QListWidgetItem>
#include <QLabel>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>

SelectRandomCreationMode::SelectRandomCreationMode(WorkingSetManager* manager,
		QWidget *parent, Qt::WindowFlags f) : QDialog(parent, f) {
	m_wellMargin = 300;

	// create ui
	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_listWidget = new QListWidget();
	m_listWidget->setSelectionMode(QListWidget::SelectionMode::MultiSelection);
	mainLayout->addWidget(m_listWidget);

	QLabel *marginLabel = new QLabel("Margin (in m):");
	mainLayout->addWidget(marginLabel);

	QDoubleSpinBox* marginSpinBox = new QDoubleSpinBox;
	marginSpinBox->setMinimum(0);
	marginSpinBox->setMaximum(std::numeric_limits<double>::max());
	marginSpinBox->setValue(m_wellMargin);
	mainLayout->addWidget(marginSpinBox);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(
				QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	mainLayout->addWidget(buttonBox);

	// fill list with wells from manager
	QList<IData*> datas = manager->folders().wells->data();
	long idx = 0;
	for (IData* data : datas) {
		if (WellHead* wellHead = dynamic_cast<WellHead*>(data)) {
			for (WellBore* well : wellHead->wellBores()) {
				m_wellBores.push_back(well);
				QListWidgetItem* item = new QListWidgetItem(well->name());
				item->setData(Qt::UserRole, QVariant((qlonglong) idx));
				m_listWidget->addItem(item);
				idx++;
			}
		}
	}

	// connects
	connect(m_listWidget, &QListWidget::itemSelectionChanged, this, &SelectRandomCreationMode::itemSelectionChanged);

	connect(marginSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
			&SelectRandomCreationMode::setValueMargin);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

SelectRandomCreationMode::~SelectRandomCreationMode() {}

QList<WellBore*> SelectRandomCreationMode::selectedWellBores() const {
	return m_selectedWellBores;
}

void SelectRandomCreationMode::itemSelectionChanged() {
	m_selectedWellBores.clear();
	for (QListWidgetItem* item : m_listWidget->selectedItems()) {
		bool ok;
		int index = item->data(Qt::UserRole).toLongLong(&ok);
		if (ok && index<m_wellBores.size() && index>=0) {
			m_selectedWellBores.push_back(m_wellBores[index]);
		}
	}
}

double SelectRandomCreationMode::wellMargin() const {
	return m_wellMargin;
}

void SelectRandomCreationMode::setValueMargin(double val) {
	m_wellMargin = val;
}

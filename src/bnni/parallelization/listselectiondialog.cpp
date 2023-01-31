#include "listselectiondialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QListWidget>
#include <QListWidgetItem>

ListSelectionDialog::ListSelectionDialog(const QStringList& list, const QString& question) {
	for (int i=0; i<list.count(); i++) {
		SelectionItem selectionItem;
		selectionItem.str = list[i];
		selectionItem.isSelected = true;
		m_list.push_back(selectionItem);
	}
	m_question = question;

	QVBoxLayout* layout = new QVBoxLayout;
	setLayout(layout);

	layout->addWidget(new QLabel(m_question));

	m_listWidget = new QListWidget;
	m_listWidget->setSelectionMode(QAbstractItemView::MultiSelection);
	for (int i=0; i<m_list.size(); i++) {
		QListWidgetItem* item = new QListWidgetItem(m_list[i].str);
		item->setSelected(m_list[i].isSelected);
		item->setData(Qt::UserRole, i);
		m_listWidget->addItem(item);
	}
	layout->addWidget(m_listWidget);

	QHBoxLayout* buttonLayout = new QHBoxLayout;
	layout->addLayout(buttonLayout);

	QPushButton* selectAllButton = new QPushButton("Select All");
	buttonLayout->addWidget(selectAllButton);
	QPushButton* deselectAllButton = new QPushButton("Deselect All");
	buttonLayout->addWidget(deselectAllButton);

	QDialogButtonBox* dialogButtonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	layout->addWidget(dialogButtonBox);

	connect(dialogButtonBox, &QDialogButtonBox::accepted, this, &ListSelectionDialog::accept);
	connect(dialogButtonBox, &QDialogButtonBox::rejected, this, &ListSelectionDialog::reject);

	connect(m_listWidget, &QListWidget::itemSelectionChanged, this, &ListSelectionDialog::selectionChanged);

	connect(selectAllButton, &QPushButton::clicked, this, &ListSelectionDialog::selectAll);
	connect(deselectAllButton, &QPushButton::clicked, this, &ListSelectionDialog::deselectAll);

	selectAll();
}

ListSelectionDialog::~ListSelectionDialog() {

}

const std::vector<ListSelectionDialog::SelectionItem> ListSelectionDialog::getList() const {
	return m_list;
}

void ListSelectionDialog::selectionChanged() {
	for (int i=0; i<m_list.size(); i++) {
		m_list[i].isSelected = false;
	}

	QList<QListWidgetItem*> selection = m_listWidget->selectedItems();
	for (int j = 0; j<selection.count(); j++) {
		bool ok;
		int i = selection[j]->data(Qt::UserRole).toInt(&ok);
		if (ok) {
			m_list[i].isSelected = true;
		}
	}
}

void ListSelectionDialog::selectAll() {
	for (int i=0; i<m_listWidget->count(); i++) {
		bool ok;
		int j = m_listWidget->item(i)->data(Qt::UserRole).toInt(&ok);
		if(ok) {
			m_list[j].isSelected = true;
		}
		m_listWidget->item(i)->setSelected(true);
	}
}

void ListSelectionDialog::deselectAll() {
	for (int i=0; i<m_listWidget->count(); i++) {
		bool ok;
		int j = m_listWidget->item(i)->data(Qt::UserRole).toInt(&ok);
		if(ok) {
			m_list[j].isSelected = false;
		}
		m_listWidget->item(i)->setSelected(false);
	}
}

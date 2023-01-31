#include "itemlistselectorwithcolor.h"

#include <QColorDialog>
#include <QHBoxLayout>
#include <QPushButton>

ItemListSelectorWithColor::ItemListSelectorWithColor(QWidget* parent, Qt::WindowFlags f): QWidget(parent, f) {
	QHBoxLayout* layout = new QHBoxLayout;
	setLayout(layout);

	m_treeWidget = new ItemListSelectorWithColorTree;
	layout->addWidget(m_treeWidget);

	connect(m_treeWidget, &QTreeWidget::itemSelectionChanged, this, &ItemListSelectorWithColor::itemSelectionChangedSlot);
}

ItemListSelectorWithColor::~ItemListSelectorWithColor() {

}

QTreeWidgetItem* ItemListSelectorWithColor::addItem(const QString& name, const QColor& color, const QVariant& userData) {
	QTreeWidgetItem* item = new QTreeWidgetItem();
	item->setText(0, name);
	item->setData(0, Qt::UserRole, userData);
	item->setData(1, Qt::UserRole, color);
	m_treeWidget->addTopLevelItem(item);

	QPushButton* button = new QPushButton;
	button->setStyleSheet(QString("QPushButton {background: %1}").arg(color.name()));

	m_treeWidget->setItemWidget(item, 1, button);
	m_buttons[item] = button;

	connect(button, &QPushButton::clicked, [this, item]() {
		QColor colorInitial = Qt::blue;
		QVariant colorVariant = item->data(1, Qt::UserRole);
		if (colorVariant.canConvert<QColor>()) {
			colorInitial = colorVariant.value<QColor>();
		}
		QColorDialog dialog(colorInitial);
		int err = dialog.exec();
		if (err==QDialog::Accepted) {
			setColor(item, dialog.selectedColor());
		}
	});

	return item;
}

void ItemListSelectorWithColor::clear() {
	m_treeWidget->clear();
	m_buttons.clear();
}

QColor ItemListSelectorWithColor::getColor(QTreeWidgetItem* item, bool* ok) {
	bool valid = item->columnCount()>=2;
	QColor color;
	if (valid) {
		QVariant v = item->data(1, Qt::UserRole);
		valid = v.canConvert<QColor>();
		if (valid) {
			color = v.value<QColor>();
		}
	}
	if (ok!=nullptr) {
		*ok = valid;
	}
	return color;
}

void ItemListSelectorWithColor::itemSelectionChangedSlot() {
	emit itemSelectionChanged();
}

QList<QTreeWidgetItem*> ItemListSelectorWithColor::selectedItems() {
	return m_treeWidget->selectedItems();
}

void ItemListSelectorWithColor::setColor(QTreeWidgetItem* item, const QColor& color) {
	item->setData(1, Qt::UserRole, color);

	auto it = m_buttons.find(item);
	if (it!=m_buttons.end()) {
		it->second->setStyleSheet(QString("QPushButton {background: %1}").arg(color.name()));
	}

	emit colorChanged(item, color);
}

ItemListSelectorWithColorTree::ItemListSelectorWithColorTree(QWidget* parent): QTreeWidget(parent) {
	setHeaderHidden(true);
	setSelectionMode(QAbstractItemView::MultiSelection);
	setStyleSheet("QTreeWidget {min-height: 3em}");
	setColumnCount(2);
}

ItemListSelectorWithColorTree::~ItemListSelectorWithColorTree() {

}


QItemSelectionModel::SelectionFlags ItemListSelectorWithColorTree::selectionCommand(const QModelIndex& index, const QEvent* event) const {
	QItemSelectionModel::SelectionFlags out;
	if (index.column()==1) {
		out = QItemSelectionModel::NoUpdate;
	} else {
		out = QTreeWidget::selectionCommand(index, event);
	}

	return out;
}

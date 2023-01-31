#ifndef SRC_WIDGET_ITEMLISTSELECTORWITHCOLOR_H
#define SRC_WIDGET_ITEMLISTSELECTORWITHCOLOR_H

#include <QTreeWidget>

class QTreeWidgetItem;
class QPushButton;

class ItemListSelectorWithColorTree;

class ItemListSelectorWithColor : public QWidget {
	Q_OBJECT
public:
	ItemListSelectorWithColor(QWidget* parent = 0, Qt::WindowFlags f = Qt::WindowFlags());
	~ItemListSelectorWithColor();

	/**
	 * add an item with 2 columns
	 * first column has DisplayRole name and UserRole userData
	 * second column is controlled by the color
	 */
	QTreeWidgetItem* addItem(const QString& name, const QColor& color, const QVariant& userData);
	void clear();
//	void removeItem(QTreeWidgetItem* item);
	QList<QTreeWidgetItem*> selectedItems();
	void setColor(QTreeWidgetItem* item, const QColor& color);

	static QColor getColor(QTreeWidgetItem* item, bool* ok=nullptr);

signals:
	void itemSelectionChanged();
	void colorChanged(QTreeWidgetItem* item, QColor color);

private slots:
	void itemSelectionChangedSlot();

private:
	std::map<QTreeWidgetItem*, QPushButton*> m_buttons;

	ItemListSelectorWithColorTree* m_treeWidget;
};

class ItemListSelectorWithColorTree : public QTreeWidget {
	Q_OBJECT
public:
	ItemListSelectorWithColorTree(QWidget* parent=0);
	~ItemListSelectorWithColorTree();
protected:
	virtual QItemSelectionModel::SelectionFlags selectionCommand(const QModelIndex& index, const QEvent* event=nullptr) const override;
};

#endif

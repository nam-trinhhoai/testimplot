#include "nvtreewidget.h"

#include <QResizeEvent>

NVTreeWidget::NVTreeWidget(QWidget* parent) : QTreeWidget(parent) {

}

NVTreeWidget::~NVTreeWidget() {

}

void NVTreeWidget::resizeEvent(QResizeEvent* event) {
	QTreeWidget::resizeEvent(event);
	QTreeWidgetItem* item = currentItem();
	if (item) {
		bool parentsExpanded = true;
		QTreeWidgetItem* parentItem = item->parent();
		int safety = 0;

		// this is a tree, there should be no loop
		while (parentsExpanded && parentItem && safety<10000) {
			parentsExpanded = parentItem->isExpanded();
			parentItem = parentItem->parent();
			safety++;
		}
		if (parentsExpanded) {
			scrollToItem(item);
		}
	}
}


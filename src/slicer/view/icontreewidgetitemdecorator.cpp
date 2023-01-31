#include "icontreewidgetitemdecorator.h"

#include <QTreeWidgetItem>

IconTreeWidgetItemDecorator::IconTreeWidgetItemDecorator(const QIcon& icon, QObject* parent) :
		ITreeWidgetItemDecorator(parent), m_icon(icon) {

}

IconTreeWidgetItemDecorator::~IconTreeWidgetItemDecorator() {

}

void IconTreeWidgetItemDecorator::decorate(QTreeWidgetItem* item, int column) {
	item->setData(column, Qt::DecorationRole, m_icon);
}

QIcon IconTreeWidgetItemDecorator::icon() const {
	return m_icon;
}

void IconTreeWidgetItemDecorator::setIcon(QIcon icon) {
	m_icon = icon;
	emit decoratorUdpated();
}

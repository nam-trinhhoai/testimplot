#include "colortreewidgetitemdecorator.h"

#include <QTreeWidgetItem>

ColorTreeWidgetItemDecorator::ColorTreeWidgetItemDecorator(const QColor& color, QObject* parent) :
		ITreeWidgetItemDecorator(parent), m_color(color) {

}

ColorTreeWidgetItemDecorator::~ColorTreeWidgetItemDecorator() {

}

void ColorTreeWidgetItemDecorator::decorate(QTreeWidgetItem* item, int column) {
	item->setData(column, Qt::DecorationRole, m_color);
}

QColor ColorTreeWidgetItemDecorator::color() const {
	return m_color;
}

void ColorTreeWidgetItemDecorator::setColor(QColor color) {
	if (color!=m_color) {
		m_color = color;
		emit decoratorUdpated();
	}
}

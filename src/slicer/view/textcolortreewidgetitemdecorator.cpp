#include "textcolortreewidgetitemdecorator.h"

#include <QTreeWidgetItem>

TextColorTreeWidgetItemDecorator::TextColorTreeWidgetItemDecorator(QObject* parent) :
		ITreeWidgetItemDecorator(parent), m_colorSet(false) {

}

TextColorTreeWidgetItemDecorator::TextColorTreeWidgetItemDecorator(const QColor& color, QObject* parent) :
		ITreeWidgetItemDecorator(parent), m_color(color), m_colorSet(true) {

}

TextColorTreeWidgetItemDecorator::~TextColorTreeWidgetItemDecorator() {

}

void TextColorTreeWidgetItemDecorator::decorate(QTreeWidgetItem* item, int column) {
	if (m_colorSet) {
		item->setData(column, Qt::ForegroundRole, m_color);
	} else {
		item->setData(column, Qt::ForegroundRole, QVariant());
	}
}

QColor TextColorTreeWidgetItemDecorator::color() const {
	return m_color;
}

void TextColorTreeWidgetItemDecorator::setColor(QColor color) {
	if (!m_colorSet || color!=m_color) {
		m_color = color;
		m_colorSet = true;
		emit decoratorUdpated();
	}
}

void TextColorTreeWidgetItemDecorator::unsetColor() {
	if (m_colorSet) {
		m_colorSet = false;
		emit decoratorUdpated();
	}
}

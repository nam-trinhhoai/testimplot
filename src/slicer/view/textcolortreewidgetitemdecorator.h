#ifndef TEXTCOLORTREEWIDGETITEMDECORATOR_H
#define TEXTCOLORTREEWIDGETITEMDECORATOR_H

#include "itreewidgetitemdecorator.h"

#include <QColor>

class TextColorTreeWidgetItemDecorator : public ITreeWidgetItemDecorator {
public:
	TextColorTreeWidgetItemDecorator(QObject* parent=0);
	TextColorTreeWidgetItemDecorator(const QColor& color, QObject* parent=0);
	~TextColorTreeWidgetItemDecorator();

	virtual void decorate(QTreeWidgetItem* item, int column=0) override;

	QColor color() const;

public slots:
	void setColor(QColor color);
	void unsetColor();

private:
	QColor m_color;
	bool m_colorSet;
};

#endif

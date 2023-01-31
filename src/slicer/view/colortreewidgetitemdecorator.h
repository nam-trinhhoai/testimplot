#ifndef COLORTREEWIDGETITEMDECORATOR_H
#define COLORTREEWIDGETITEMDECORATOR_H

#include "itreewidgetitemdecorator.h"

#include <QColor>

class ColorTreeWidgetItemDecorator : public ITreeWidgetItemDecorator {
public:
	ColorTreeWidgetItemDecorator(const QColor& color, QObject* parent=0);
	~ColorTreeWidgetItemDecorator();

	virtual void decorate(QTreeWidgetItem* item, int column=0) override;

	QColor color() const;

public slots:
	void setColor(QColor color);

private:
	QColor m_color;
};

#endif

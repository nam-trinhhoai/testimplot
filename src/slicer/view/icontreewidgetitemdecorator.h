#ifndef ICONTREEWIDGETITEMDECORATOR_H
#define ICONTREEWIDGETITEMDECORATOR_H

#include "itreewidgetitemdecorator.h"

#include <QIcon>

class IconTreeWidgetItemDecorator : public ITreeWidgetItemDecorator {
public:
	IconTreeWidgetItemDecorator(const QIcon& icon, QObject* parent=0);
	~IconTreeWidgetItemDecorator();

	virtual void decorate(QTreeWidgetItem* item, int column=0) override;

	QIcon icon() const;

public slots:
	void setIcon(QIcon icon);

private:
	QIcon m_icon;
};

#endif

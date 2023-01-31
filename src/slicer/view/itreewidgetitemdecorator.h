#ifndef ITREEWIDGETITEMDECORATOR_H
#define ITREEWIDGETITEMDECORATOR_H

#include <QObject>

class QTreeWidgetItem;

class ITreeWidgetItemDecorator : public QObject {
	Q_OBJECT
public:
	ITreeWidgetItemDecorator(QObject* parent=0);
	virtual ~ITreeWidgetItemDecorator();

	virtual void decorate(QTreeWidgetItem* item, int column=0) = 0;

signals:
	void decoratorUdpated();
};

#endif // ITREEWIDGETITEMDECORATOR_H

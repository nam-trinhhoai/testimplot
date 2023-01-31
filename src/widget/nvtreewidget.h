#ifndef SRC_WIDGET_NVTREEWIDGET_H
#define SRC_WIDGET_NVTREEWIDGET_H

#include <QTreeWidget>

class NVTreeWidget : public QTreeWidget{
	Q_OBJECT
public:
	NVTreeWidget(QWidget* parent=nullptr);
	~NVTreeWidget();

protected:
	virtual void resizeEvent(QResizeEvent* event) override;
};

#endif

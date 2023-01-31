#ifndef DockWidgetSizeGrid_H
#define DockWidgetSizeGrid_H

#include <QSizeGrip>

class DockWidgetSizeGrid : public QSizeGrip{
	Q_OBJECT
public:
	DockWidgetSizeGrid(QWidget *parent);
	virtual ~DockWidgetSizeGrid();

	 void mouseMoveEvent(QMouseEvent * event) override;

signals:
	 	void geometryChanged(const QRect & geom);
private:
	 QWidget * sizegrip_topLevelWidget(QWidget* w);


};

#endif

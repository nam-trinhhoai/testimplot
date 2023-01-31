#ifndef View3DGraphicsView_H
#define View3DGraphicsView_H

#include "monotypegraphicsview.h"
class QSpinBox;

class View3DGraphicsView: public MonoTypeGraphicsView {
Q_OBJECT
public:
	View3DGraphicsView(WorkingSetManager *factory, QString uniqueName, QWidget *parent);
	virtual ~View3DGraphicsView();
private slots:
	void zScaleChanged(int val);
private:
	QWidget* createZScaleWidget();
private:
	QSpinBox *m_zscale;
};

#endif

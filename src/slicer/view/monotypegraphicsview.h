#ifndef MonoTypeGraphicsView_H
#define MonoTypeGraphicsView_H

#include "abstractgraphicsview.h"

class MonoTypeGraphicsView: public AbstractGraphicsView {
Q_OBJECT
public:
	MonoTypeGraphicsView(WorkingSetManager *factory, ViewType viewType, QString uniqueName, QWidget *parent);
	virtual ~MonoTypeGraphicsView();

public slots:
	void addView();

private:
	ViewType m_viewType;
};

#endif

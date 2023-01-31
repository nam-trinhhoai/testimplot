#ifndef BaseMapGraphicsView_H
#define BaseMapGraphicsView_H

#include "mousesynchronizedgraphicsview.h"

class BaseMapGraphicsView: public MouseSynchronizedGraphicsView {
Q_OBJECT
public:
	BaseMapGraphicsView(WorkingSetManager *factory, QString uniqueName, QWidget *parent);
	virtual ~BaseMapGraphicsView();
};

#endif

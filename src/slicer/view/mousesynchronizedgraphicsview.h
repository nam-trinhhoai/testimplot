#ifndef MouseSynchronizedGraphicsView_H
#define MouseSynchronizedGraphicsView_H

#include <QMutex>
#include "monotypegraphicsview.h"

class MouseSynchronizedGraphicsView: public MonoTypeGraphicsView {
Q_OBJECT
public:
	MouseSynchronizedGraphicsView(WorkingSetManager *factory,ViewType viewType, QString uniqueName,QWidget *parent);
	virtual ~MouseSynchronizedGraphicsView();
protected slots:
	void viewPortChanged(const QPolygonF &poly);
	void unregisterView(AbstractInnerView *toBeDeleted) override;
protected:
	void registerView(AbstractInnerView *newView) override;
private:
	ViewType m_viewType;
	QMutex m_mutexView;
};

#endif

#ifndef SplittedView_H
#define SplittedView_H

#include <QMouseEvent>
#include <QWidget>
#include <QMutex>
#include <QPushButton>
#include <QList>
#include <QMap>

#include <kddockwidgets/MainWindow.h>
#include "viewutils.h"
#include "abstractgraphicsview.h"

class MouseTrackingEvent;
class QGLCrossItem;

class AbstractInnerView;
class AbstractGraphicRep;
class QDockWidget;
class QTabWidget;
class QGraphicsItem;
class QString;

typedef enum viewMode {
	eTypeTabMode = 0,
	eTypeSplitMode
}eViewMode;

class SplittedView: public KDDockWidgets::MainWindow {
	Q_OBJECT
public:
	static QString viewModeLabel( eViewMode e);
	SplittedView(ViewType v,QList<AbstractGraphicRep*> repList,eViewMode eMode,AbstractInnerView* parent);
	virtual ~SplittedView();
	void showRep();
	void addView(QList<AbstractGraphicRep*>repList,AbstractGraphicRep* pRep,ViewType v);
	AbstractInnerView* createInnerView(ViewType v,AbstractGraphicRep *pRep);
	QList<AbstractInnerView*> getInnerViews();
public slots:
	void innerViewMouseMoved(MouseTrackingEvent *event);
	void viewPortChanged(const QPolygonF &poly);
	void unregisterView(AbstractInnerView * pInnerView);
	void changeviewMode();
	void geometryChanged(AbstractInnerView *newView,const QRect & geom);

signals:
	void viewMouseMoved(MouseTrackingEvent *event);
private:
	void tabView();
	void splitView();

	QMap <AbstractInnerView *,QList<AbstractGraphicRep*>> m_InnerViews;
	QList <AbstractGraphicRep *> m_Rep;
	QList<QGraphicsItem*>  m_dynamicItems;
	AbstractInnerView *m_parent;
	int m_NbColumn;
	eViewMode m_eViewMode;
	QPushButton *m_viewMode;
	QMutex m_mutexView;
};

#endif

#ifndef AbstractGraphicsView_H
#define AbstractGraphicsView_H

#include <QMouseEvent>
#include <QWidget>
#include <QList>
#include "pickinginfo.h"
#include <kddockwidgets/MainWindow.h>
#include "viewutils.h"

#include "SynchroMultiView.h"

class QGraphicsView;
class QMdiArea;
class QGraphicsScene;
class QGLScaleBarItem;
class IGraphicRepFactory;
class AbstractGraphicRep;
class QVBoxLayout;
class DataControler;
class QGraphicsObject;
class QGLLineItem;
class IDataControlerHolder;
class QAbstractItemModel;
class QTreeWidgetItem;
class QTreeWidget;
class WorkingSetManager;
class IData;
class QSplitter;
class MouseTrackingEvent;
class QGLCrossItem;
class BaseQGLGraphicsView;
class AbstractInnerView;
class QMdiSubWindow;
class QGridLayout;
class QHBoxLayout;
class QToolBar;
class QDockWidget;
class QTabWidget;

class RandomLineView;
class WellBore;

typedef enum {
    eTypeStandard = 0,
    eTypeOrthogonal,
}eRandomType;

class AbstractGraphicsView: public KDDockWidgets::MainWindow {
Q_OBJECT
public:
	AbstractGraphicsView(WorkingSetManager *factory, QString uniqueName,
	QWidget *parent);
	virtual ~AbstractGraphicsView();

	//The controlers this view is holding
	QList<DataControler*> getControlers() const;
	void closeEvent(QCloseEvent * closeEvent) override;

signals:
	void isClosing(AbstractGraphicsView * win);

	void controlerActivated(DataControler*);
	void controlerDesactivated(DataControler*);

	void viewMouseMoved(MouseTrackingEvent *event);
public slots:
	//Controler comming from other view
	void addExternalControler(DataControler *controler);
	void removeExternalControler(DataControler *controler);

	void externalMouseMoved(MouseTrackingEvent * event);
protected slots:
	void geometryChanged(const QRect & geom);
	void geometryChanged(AbstractInnerView *newView,const QRect & geom);
	virtual void unregisterView(AbstractInnerView *newView);

	void subWindowActivated(AbstractInnerView *window);
	void askToSplit(AbstractInnerView * toSplit,ViewType type, bool restictToMonoTypeSplitChild, Qt::Orientation orientation);

	void showParameters();
	void arrangeWindows();

	void onCustomContextMenu(const QPoint &point);
	void itemSelectionChanged();
	void dataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight,
			const QVector<int> &roles);

	void resetZoom();

	void innerViewControlerActivated(DataControler*c);
	void innerViewControlerDesactivated(DataControler*c);

	void innerViewMouseMoved(MouseTrackingEvent *event);

protected:
	virtual void registerView(AbstractInnerView *newView);

	AbstractInnerView *generateView(ViewType viewType, bool restrictedToMonoTypeSplit);

	QToolBar * toolBar()const;
	QWidget * createLateralButtonWidget();
	virtual void createToolbar();

	QVector<AbstractInnerView *> innerViews() const;

	bool event(QEvent *event) override;

	void addExternalControler(DataControler *controler,
			QTreeWidgetItem *parentItem);
	void removeExternalControler(DataControler *controler,
			QTreeWidgetItem *parentItem);

	void showPropertyPanel(AbstractGraphicRep *rep);
	void releasePropertyPanel();

	virtual void showRep(AbstractGraphicRep *rep);
	virtual void hideRep(AbstractGraphicRep *rep);
	void hideRgtRep(AbstractGraphicRep *rep);

	void registerWindowControlers(AbstractInnerView *newView,
			AbstractInnerView *existingView);
	void unregisterWindowControlers(AbstractInnerView *toBeDeleted,
			AbstractInnerView *existingView);


	void askToSplitStep2(AbstractInnerView* newView, AbstractInnerView * toSplit,ViewType type, bool restictToMonoTypeSplitChild, Qt::Orientation orientation);
	RandomLineView* createRandomView(QPolygonF polygon, AbstractInnerView * toSplit, bool restictToMonoTypeSplitChild, Qt::Orientation orientation);
	void selectRandomActionWithUI(AbstractInnerView * toSplit, bool restictToMonoTypeSplitChild, Qt::Orientation orientation);
	void addRandomFromWellBore(QList<WellBore*> well, double margin, AbstractInnerView * toSplit, bool restictToMonoTypeSplitChild, Qt::Orientation orientation);
protected:
	QList<DataControler*> m_externalControler;

	KDDockWidgets::DockWidget *m_parametersControler;

	QAction *m_parameterAction;
	QToolBar * m_toolbar;
	QSplitter *m_parameterSplitter;
	QTabWidget *m_controlTabWidget;

	//Tree Handling
	QTreeWidgetItem *m_rootItem;
	QAbstractItemModel *m_model;
	QTreeWidget *m_treeWidget;

	//The root manager of the different entities
	WorkingSetManager *m_currentManager;

	std::size_t getNewUniqueId();
	void changeViewName(AbstractInnerView* view, QString newName);

	SynchroMultiView synchroMultiView;

private:
	QWidget* getPlaceHolderForUse();
	void releasePlaceHolder();

	std::size_t m_uniqueId = 0; // use to name new DockWidget

	QWidget* m_placeHolderWidget = nullptr;

	AbstractGraphicRep* m_lastPropPanelRep = nullptr;
};

struct WidgetPoperTrait {
public:
	virtual ~WidgetPoperTrait() {};

	//this method is meant to show a widget in the current context
	//Some client ask to show the given widget and the parent decides if it can display the widget or need to be given the parent again (loop on this step while needed)
	virtual void popWidget(QWidget* widget) = 0;
};

#endif

#ifndef AbstractInnerView_H
#define AbstractInnerView_H

#include <QList>
#include <QRectF>
#include <QLabel>
#include <QDockWidget>
#include <kddockwidgets/DockWidget.h>
#include <kddockwidgets/Config.h>
#include "viewutils.h"

class DataControler;
class AbstractGraphicRep;
class PickingTask;
class MouseTrackingEvent;
class QPushButton;
class QHBoxLayout;
class QVBoxLayout;
class QMenu;
class QActionGroup;
class IsoSurfaceBuffer;

typedef enum {
	eModeStandardView=0,
	eModeSplitView
}eModeView;

class AbstractInnerView: public KDDockWidgets::DockWidget {
Q_OBJECT
public:
	AbstractInnerView(bool restictToMonoTypeSplit, QString uniqueName ,eModeView typeView=eModeStandardView);
	~AbstractInnerView();

	void setViewIndex(int val);
	void setDefaultTitle(QString title); // use old title creation process if title is null or empty

	ViewType viewType() const;

	//The controlers this view is holding
	QList<DataControler*> getControlers() const;

	virtual void showRep(AbstractGraphicRep *rep);
	virtual void hideRep(AbstractGraphicRep *rep);

	void registerPickingTask(PickingTask *task);
	void unregisterPickingTask(PickingTask *task);

	virtual void resetZoom()=0;

	AbstractGraphicRep * lastRep();
	const QList<AbstractGraphicRep*>& getVisibleReps() const;

    void enterEvent(QEnterEvent *event) override;


	QString defaultTitle() const {
		return m_defaultTitle;
	}

	QString suffixTitle() const;

	QString getBaseTitle() const;

	QList<AbstractGraphicRep*> visibleReps() {
		return m_visibleReps;
	}


	IsoSurfaceBuffer getHorizonBuffer();

	QString GraphicsLayersDirPath();

signals:
	void viewEnter(AbstractInnerView * view);
	void askToSplit(AbstractInnerView * toSplit,ViewType type, bool restictToMonoTypeSplitChild, Qt::Orientation orientation);
	void askGeometryChanged(AbstractInnerView * toSplit,const QRect & geom);

	void repAdded(AbstractGraphicRep*);
	void controlerActivated(DataControler*);
	void controlerDesactivated(DataControler*); // if needed create new signal controlerDestroyed to avoid crashes

	void isClosing(AbstractInnerView * win);

	void viewMouseMoved(MouseTrackingEvent *event);

	void sliceChangedFromView(int value, int delta, AbstractInnerView* abstractInnerView);  //GS

public slots:
	virtual void addExternalControler(DataControler *controler);
	virtual void removeExternalControler(DataControler *controler);

	virtual void externalMouseMoved(MouseTrackingEvent *event)=0;

	virtual void splitDockWidgetMulti();
	virtual void showPalette();

protected slots:
	void geometryChanged(const QRect & geom);
protected:
	QPushButton* getPaletteButton();
	QPushButton* getSplitButton();

	QWidget *generateSizeGrip();

	//Those function handle the mouse cursor converions between a local view CRS and the aboslute one
	virtual bool absoluteWorldToViewWorld(MouseTrackingEvent &event)=0;
	virtual bool viewWorldToAbsoluteWorld(MouseTrackingEvent &event)=0;

    void closeEvent(QCloseEvent *event) override;

    virtual void updateTile(const QString &name);

    QPair<QMenu *,QActionGroup *> generateViewMenu();

    QPushButton * createTitleBarButton(const QString & iconPath, const QString & tooltip) const;

    void emitSplit(int value,Qt::Orientation orientation);

	virtual void cleanupRep(AbstractGraphicRep *rep);
protected:
	ViewType m_viewType;
	bool m_restictToMonoTypeSplitChild;

	int m_currentViewIndex;

	//Caching
	QList<AbstractGraphicRep*> m_visibleReps;

	//Containing all the controler coming from other viewer
	QList<DataControler*> m_externalControler;
	QVector<PickingTask*> m_pickingTask;

	QVBoxLayout *m_NameLayout;
	QString m_defaultTitle = "";
	QString m_suffixTitle;
	QMap<AbstractGraphicRep*, QMetaObject::Connection> m_destructionConnectionMap;
	QMap<AbstractGraphicRep*, DataControler*> m_repToControlerMap; // only for reps that implement IDataControlerProvider interface

	eModeView m_modeView;
};

#endif


#ifndef SlicerWindow_H
#define SlicerWindow_H

#include <QMainWindow>
#include <QList>

#include "mousetrackingevent.h"
#include "seismic3dabstractdataset.h"

class IData;
class AbstractGraphicsView;
class DataControler;
class GraphicsView;
class WorkingSetManager;
class AbstractGraphicRep;
class SeismicSurvey;
class QOpenGLContext;

class QListWidget;
class MouseTrackingEvent;
class QSystemTrayIcon;

class SlicerWindow: public QMainWindow {
Q_OBJECT
public:
	SlicerWindow(QString uniqueName, QWidget *parent = 0);
	virtual ~SlicerWindow();

	static SlicerWindow* get();
public:
	void open(const std::string &seismicPath, const std::string &rgtPath,
			bool forceCPU);
	void openSismageProject(const QString &path);

	QVector<AbstractGraphicsView*> currentViewerList() const;
protected slots:
	void open();
	void openSismageProject();
	void openManager();

	void openBaseMapWindow();
	void openInlineWindow();
	void openXlineWindow();
	void openView3DWindow();
	void openMultiViewWindow();

	void isClosing(AbstractGraphicsView *w);

	void dataAdded(IData *d);
	void dataRemoved(IData *d);

	void viewMouseMoved(MouseTrackingEvent *event);

private:
	QVector<AbstractGraphicsView*> graphicsView();
	QWidget* createWorkingSetView();

	void registerWindowControlers(AbstractGraphicsView *newView,
			AbstractGraphicsView *existingView);
	void unregisterWindowControlers(AbstractGraphicsView *toBeDeleted,
			AbstractGraphicsView *existingView);

	void registerWindow(AbstractGraphicsView *w);

	Seismic3DAbstractDataset* appendDataset(SeismicSurvey *baseSurvey,
			const std::string &path, bool forceCPU);

	void createTrayActions();
	void createTrayIcon();

	QString getNewUniqueNameForView();

private:

	static SlicerWindow *m_mainWindow;

	QMenu *m_viewMenu;

	QVector<AbstractGraphicsView*> m_registredViewers;
	WorkingSetManager *m_manager;
	QListWidget *m_listView;

	QAction *m_basemapAction;
	QAction *m_inlineAction;
	QAction *m_xlineAction;
	QAction *m_qt3dAction;
	QAction *m_multiViewAction;

	QAction *m_minimizeAction;
	QAction *m_maximizeAction;
	QAction *m_restoreAction;
	QAction *m_quitAction;

	QSystemTrayIcon *m_trayIcon;
	QMenu *m_trayIconMenu;

	// To generate unique names for views even if there are 2 slicer windows
	std::size_t m_uniqueId = 0;
	QString m_uniqueName;

};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_MAINWIDGET_H_ */

#ifndef MultiView_H
#define MultiView_H

#include <QQuickView>
#include <QQmlEngine>
#include <QQmlContext>
#include "scenemultimanager.h"

#include <kddockwidgets/DockWidget.h>

class MultiView: public KDDockWidgets::DockWidget
{
Q_OBJECT
	public:

		MultiView(bool restictToMonoTypeSplit,QString uniqueName,QWidget *parent=nullptr);
		virtual ~MultiView();

public slots:
		void onQMLReady(QQuickView::Status status);

	private:
		QQuickView *m_quickview = nullptr;

		SceneMultiManager* m_sceneMultiManager = nullptr;
};

#endif

#ifndef BaseMapWindow_H
#define BaseMapWindow_H

#include <QMainWindow>

#include "lookuptable.h"

class QProgressBar;
class QLabel;
class QAction;
class BaseMapQGLGraphicsView;
class QGraphicsScene;
class PaletteWidget;

class QGLGDALFullImage;
class QGLFullImageItem;

class QGLTiledImageItem;
class QGLGDALTiledImage;


class BaseMapWindow : public QMainWindow
{
	Q_OBJECT
public:
	BaseMapWindow(QWidget *parent=0);
	~BaseMapWindow();

protected slots:
	void openSimpleImage();
	void openShapefile();

	void mouseMoved(const QRectF& visibleArea, double worldX,double worldY);
	void refresh();
protected:
	void createActions();
    void createMenus();

    BaseMapQGLGraphicsView* m_view;
	QGraphicsScene* m_scene;

	QAction *m_openAction;
	QAction *m_openShapefileAction;
	QAction *m_exitAction;
	QMenu* m_fileMenu;

	PaletteWidget * m_palette;

	QGLFullImageItem *m_item;
	QGLGDALFullImage *m_image;


//	QGLTiledImageItem *m_item;
//	QGLGDALTiledImage *m_image;

};

#endif // MAINWINDOW_H

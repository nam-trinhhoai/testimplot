#ifndef Abstract2DInnerView_H
#define Abstract2DInnerView_H

#include "abstractinnerview.h"
#include "pickinginfo.h"
#include "geotimegraphicsview.h"
#include "idepthview.h"
#include "PointCtrl.h"
#include <kddockwidgets/MainWindow.h>
#include <QContextMenuEvent>
#include <QMenu>


class QGridLayout;
class QVBoxLayout;
class QGraphicsScene;
class BaseQGLGraphicsView;
class QGLScaleBarItem;
class QGLCrossItem;
class QGLLineItem;
class BaseQGLGraphicsView;
class StatusBar;
class QToolButton;
class QGraphicsItem;
class IData;
class AbstractGraphicsView;
class GraphEditor_RegularBezierPath;
class GraphEditor_LineShape;
class MtLengthUnit;
class PointCtrl;

class Abstract2DInnerView: public AbstractInnerView, public IDepthView {
Q_OBJECT
public:
	Abstract2DInnerView(bool restictToMonoTypeSplit,BaseQGLGraphicsView *view,
			BaseQGLGraphicsView *verticalAxisView,
			BaseQGLGraphicsView *horizontalAxisView, QString uniqueNamee,eModeView typeView=eModeStandardView,
			KDDockWidgets::MainWindow * geoTimeView=nullptr);

	~Abstract2DInnerView();

	virtual void showRep(AbstractGraphicRep *rep) override;
	virtual void hideRep(AbstractGraphicRep *rep) override;

	void resetZoom() override;

	void setViewRect(const QRectF & viewArea);
	QPolygonF viewRect() const;

	QPointF getLastMousePosition() const {
		return m_lastMousePosition;
	}

	QRectF worldBounds() const {
		return m_worldBounds;
	}
	bool wordBoundsInitialized() const {
		return m_wordBoundsInitialized;
	}

	QGraphicsScene* scene() const {
		return m_scene;
	}

	BaseQGLGraphicsView* view() const {
		return m_view;
	}

	KDDockWidgets::MainWindow * geotimeView()
	{
		return m_GeoTimeView;
	}

	QPointF ConvertToImage(QPointF point);
	void deleteData (QGraphicsItem*);
	static int getPickingItemZ();

	QList<IData *> detectWellsIncludedInItem(QGraphicsItem *item);
	void deselectWellsIncludedInItem(QGraphicsItem *item);

	void setNurbsPoints(QVector<PointCtrl> listeCtrls,GraphEditor_ListBezierPath* path , QVector<QPointF>  listepoints,QString name,bool isopen =true, bool withTangent = false,QColor col=Qt::yellow);
	void setNurbsPoints(QVector<QPointF>  listepoints,bool isopen =true, bool withTangent = false,QColor col =Qt::yellow);
	void refreshNurbsPoints(QVector<QPointF>  listepoints, bool isopen = true, bool withTangent = false,QColor col = Qt::yellow);
	void refreshNurbsPoints(QVector<PointCtrl> listeCtrls,QVector<QPointF>  listepoints,bool isopen,bool withTangent,QPointF cross,QColor col);
	void refreshNurbsPoints(GraphEditor_ListBezierPath* path, QColor col);

	void showRandomView(bool isOrtho,GraphEditor_LineShape* line, RandomLineView * randomOrtho,QString name);
	void showRandomView(bool isOrtho,QVector<QPointF>  listepoints);

	void randomLineDeleted(RandomLineView* random);


	void setNameItem(QString name);
	void setNurbsSelected(QString name);
	void setNurbsDeleted(QString name);


	void deleteGeneratrice(QString);
	void directriceDeleted(QString );

	//void refreshOrtho(QVector3D);

	// Should only be used for depth. Time should not take this into account
	// Only used for depth value display purposes
	// depth internal computation are in meter
	virtual const MtLengthUnit* depthLengthUnit() const override;

signals:
	//void updateOrthoFrom3D(QVector3D);
	void selectedNurbs(QString);
	void deletedNurbs(QString);
//	void deletedDirectriceNurbs(QString);

	void deletedGeneratrice(QString);
	void signalRandomView(bool isOrtho,QVector<QPointF>  listepoints);
	void signalRandomView(bool isOrtho,GraphEditor_LineShape* line, RandomLineView * randomOrtho,QString name );
	void signalRandomViewDeleted(RandomLineView*);
	void addNurbsPoints(QVector<QPointF>  listepoints,bool withTangent,GraphEditor_ListBezierPath* path ,QString name, QColor col);
	void updateNurbsPoints(QVector<QPointF>  listepoints,bool withTangent,QColor col);
	void updateNurbsPoints(GraphEditor_ListBezierPath* path,QColor );
	void addCrossPoints(QVector<PointCtrl> listeCtrls,QVector<QPointF> listepoints,bool isopen,QPointF);
	void addCrossPoints(QVector<QPointF> listepoints,bool isopen);
	void addCrossPoints(GraphEditor_ListBezierPath* path);
	void viewAreaChanged(const QPolygonF & pos);
	void contextualMenuSignal(Abstract2DInnerView* emitingView, double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& mainMenu);

public slots:
	void addExternalControler(DataControler *controler) override;
	void removeExternalControler(DataControler *controler)override;

	void externalMouseMoved(MouseTrackingEvent *event) override;

	void scrollBarPosChanged();
	virtual void setDepthLengthUnit(const MtLengthUnit* depthLengthUnit) override;
protected slots:
	void mouseMoved(double worldX, double worldY, Qt::MouseButton button,
			Qt::KeyboardModifiers keys);
	void mousePressed(double worldX, double worldY, Qt::MouseButton button,
			Qt::KeyboardModifiers keys);
	void mouseRelease(double worldX, double worldY, Qt::MouseButton button,
			Qt::KeyboardModifiers keys);
	void mouseDoubleClick(double worldX, double worldY, Qt::MouseButton button,
                        Qt::KeyboardModifiers keys);
	void contextMenu(double worldX, double worldY, QContextMenuEvent::Reason reason,
                        QMenu& menu);

	void scaleChanged(double sx, double sy);

	virtual bool updateWorldExtent(const QRectF &worldExtent);

	void startGraphicToolsDialog();
	void saveGraphicLayer();
	void loadCultural();

protected:
	QToolButton* createToogleBarButton(const QString &iconPath,
			const QString &tooltip) const;

	void connnectScrollBarToExternalViewAreaChangedSignal();
	void disconnnectScrollBarToExternalViewAreaChangedSignal();


	StatusBar* statusBar() const;

	void resizeEvent(QResizeEvent * resizeEvent) override;

	virtual void contextualMenuFromGraphics(double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& mainMenu) = 0;

	virtual void cleanupRep(AbstractGraphicRep *rep) override;

	virtual void setDepthLengthUnitProtected(const MtLengthUnit* depthLengthUnit);

	// take ownership of widget, if there is a previous widget it will be deleted
	void setScenesTopCornerWidget(QWidget* widget);

private:
	//QString GraphicsLayersDirPath();
	void propagateMouseMoveEvent(double worldX, double worldY,
			Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
	QVector<PickingInfo> collectPickInfo(double worldX, double worldY);
	virtual bool fillStatusBar(double worldX, double worldY,
			MouseTrackingEvent &event);

	void showControler(DataControler *controler, AbstractGraphicRep *rep);
	void releaseControler(DataControler *controler, AbstractGraphicRep *rep);
	int getZFromRep(AbstractGraphicRep* rep);

protected:
	static int SCALE_ITEM_Z;
	static int DATA_ITEM_Z;
	static int COURBE_ITEM_Z;
	static int CROSS_ITEM_Z;
	static int PICKING_ITEM_Z;

	static int HORIZONTAL_AXIS_SIZE;
	static int VERTICAL_AXIS_SIZE;

	QRectF m_worldBounds;
	bool m_wordBoundsInitialized;

	BaseQGLGraphicsView *m_view;
	QGraphicsScene *m_scene;

	BaseQGLGraphicsView *m_verticalAxisView;
	QGraphicsScene *m_verticalAxisScene;

	BaseQGLGraphicsView *m_horizontalAxisView;
	QGraphicsScene *m_horizontalAxisScene;

	QGLScaleBarItem *m_scaleItem;
	QGLCrossItem *m_crossHairItem;


	StatusBar *m_statusBar;

	QVBoxLayout *m_mainLayout;

	QPointF m_lastMousePosition;
	KDDockWidgets::MainWindow  *m_GeoTimeView;

	const MtLengthUnit* m_depthLengthUnit;

private:
	QGridLayout* m_box;
	QWidget* m_boxTopCornerPreviousWidget = nullptr;
};

#endif


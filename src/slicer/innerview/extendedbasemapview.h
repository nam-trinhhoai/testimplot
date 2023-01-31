#ifndef ExtendedBaseMapView_H
#define ExtendedBaseMapView_H

#include "abstract2Dinnerview.h"
#include <kddockwidgets/MainWindow.h>
class QGLGridItem;
class QGLGridAxisItem;
class RulerPicking;
class GeRectangle;
class GeEllipse;
class GePolygon;

class QSlider;
class QSpinBox;
class QLabel;
class AbstractGraphicsView;
class NurbsWidget;

class ExtendedBaseMapView : public Abstract2DInnerView{
	  Q_OBJECT
public:
	enum viewMode {
		eTypeStackMode = 0,
		eTypeTabMode,
		eTypeSplitMode
	} /*eModeView*/;

private:
	static QString viewModeLabel(viewMode e);

public:
	ExtendedBaseMapView(bool restictToMonoTypeSplit,QString uniqueName,
			viewMode typeView=eTypeStackMode,
			KDDockWidgets::MainWindow* geoTimeView=nullptr);
	virtual ~ExtendedBaseMapView();

	double getWellMapWidth() const;
	void setWellMapWidth(double value);

public slots:
	virtual void updateStackIndex(int stackIndex);

signals:
	void signalWellMapWidth(double);

protected:
	bool updateWorldExtent(const QRectF& worldExtent) override;

	virtual bool absoluteWorldToViewWorld(MouseTrackingEvent &event) override;
	virtual bool viewWorldToAbsoluteWorld(MouseTrackingEvent &event) override;

	void showRep(AbstractGraphicRep *rep) override;
	void hideRep(AbstractGraphicRep *rep) override;
	void cleanupRep(AbstractGraphicRep *rep) override;

	void updateAxisExtent(const QRectF &worldExtent);

	virtual void contextualMenuFromGraphics(double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& mainMenu) override;

protected slots:
	void startRuler(bool checked);
	void startCreateNurbs(bool checked);
//	void startRectanglePicking(bool checked);
//	void startEllipsePicking(bool checked);
//	void startPolygonPicking(bool checked);
	void recomputeRange();
	void export2Sismage();
	void exportMultiLayer2Sismage();

private:
	void defineStackMinMax(const QVector2D &imageMinMax,
			int step);
	void defineStackVal(int image);
	QWidget* createStackBox(const QString &title);
#if 1
	QWidget* createViewModeSelector();
#endif
	static int GRID_ITEM_Z;
	QGLGridItem *m_baseGridItem;

	QGLGridAxisItem * m_verticalAxis;
	QGLGridAxisItem * m_horizontalAxis;

	RulerPicking* m_rulerPicking = nullptr;

	int m_currentStack;
	double m_contextualWorldX = -1;
	double m_contextualWorldY = -1;
	bool m_isRulerOn = false;
	bool m_isRectanglePickingOn = false;
	bool m_isEllipsePickingOn = false;
	bool m_isPolygonPickingOn = false;
	GeRectangle* m_geRectangle = nullptr;
	GeEllipse* m_geEllipse = nullptr;
	GePolygon* m_gePolygon = nullptr;
	QSlider* m_stackSlider = nullptr;
	QSpinBox* m_stackSpin = nullptr;
	QLabel* m_stackLabel = nullptr;

	// common value for wells
	double m_wellMapWidth = 2.0;

	std::map<AbstractGraphicRep*, QVector2D> m_repToRange;
};

#endif

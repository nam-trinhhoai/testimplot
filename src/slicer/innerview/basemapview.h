#ifndef BaseMapView_H
#define BaseMapView_H

#include "abstract2Dinnerview.h"
class QGLGridItem;
class QGLGridAxisItem;
class RulerPicking;
class GeRectangle;
class GeEllipse;
class GePolygon;
class GeRectanglePicking;
class AbstractGraphicsView;


class BaseMapView : public Abstract2DInnerView{
	  Q_OBJECT
public:
	BaseMapView(bool restictToMonoTypeSplit,QString uniqueName,eModeView typeView=eModeStandardView,
			AbstractGraphicsView* geoTimeView=nullptr);
	virtual ~BaseMapView();

	double getWellMapWidth() const;
	void setWellMapWidth(double value);

signals:
	void signalWellMapWidth(double);

protected:
	bool updateWorldExtent(const QRectF& worldExtent) override;

	virtual bool absoluteWorldToViewWorld(MouseTrackingEvent &event) override;
	virtual bool viewWorldToAbsoluteWorld(MouseTrackingEvent &event) override;

	void showRep(AbstractGraphicRep *rep) override;

	void updateAxisExtent(const QRectF &worldExtent);

	virtual void contextualMenuFromGraphics(double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& mainMenu) override;

protected slots:
	void startRuler(bool checked);


//	void startRectanglePicking(bool checked);
//	void startEllipsePicking(bool checked);
//	void startPolygonPicking(bool checked);
	void export2Sismage();
	void exportMultiLayer2Sismage();
	void computeTmap();

private:
	static int GRID_ITEM_Z;
	QGLGridItem *m_baseGridItem;

	QGLGridAxisItem * m_verticalAxis;
	QGLGridAxisItem * m_horizontalAxis;

	RulerPicking* m_rulerPicking = nullptr;

	double m_contextualWorldX = -1;
	double m_contextualWorldY = -1;
	bool m_isRulerOn = false;
	bool m_isRectanglePickingOn = false;
	bool m_isEllipsePickingOn = false;
	bool m_isPolygonPickingOn = false;
	GeRectangle* m_geRectangle = nullptr;
	GeEllipse* m_geEllipse = nullptr;
	GePolygon* m_gePolygon = nullptr;
	GeRectanglePicking* m_rectanglePicking = nullptr;

	// common value for wells
	double m_wellMapWidth = 2.0;
};

#endif

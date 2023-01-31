#ifndef WellBoreLayerOnSlice_H
#define WellBoreLayerOnSlice_H

#include "graphiclayer.h"

class QGraphicsView;
class QGraphicsPathItem;
class QGraphicsItem;
class QEvent;
class WellBoreRepOnSlice;

class WellBoreLayerOnSlice : public GraphicLayer{
	  Q_OBJECT
public:
	  WellBoreLayerOnSlice(WellBoreRepOnSlice *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent);
	virtual ~WellBoreLayerOnSlice();

	virtual void show() override;
	virtual void hide() override;

	virtual QRectF boundingRect() const override;
	virtual void refresh() override;
	void refreshLog();

	bool isShown() const;

	void setLogColor(QColor color);
	void toggleLogDisplay(bool showLog);

	double origin() const;
	void setOrigin(double val);
	double width() const;
	void setWidth(double val);
	double logMin() const;
	void setLogMin(double val);
	double logMax() const;
	void setLogMax(double val);
	void setPenWidth(double val);

signals:
	void layerShownChanged(bool toggle);

//protected:
//	bool eventFilter(QObject* watched, QEvent* ev) override;

private:
	WellBoreRepOnSlice *m_rep;

	QGraphicsView* m_view;
	QGraphicsPathItem* m_item;
	QGraphicsPathItem* m_itemLog;
	bool m_isShown=false;
	bool m_isShownLog=false;

	double m_origin;
	double m_width;
	double m_logMin;
	double m_logMax;

	double m_cachedScale = 1;
	bool m_scaleInit = false;
};

#endif

/*
 * Canvas2dFullSync.h
 *
 *  Created on: 15 janv. 2018
 *      Author: j0483271
 */

#ifndef MURATAPP_SRC_VIEW_CANVAS2D_CANVAS2DFULLSYNC_H_
#define MURATAPP_SRC_VIEW_CANVAS2D_CANVAS2DFULLSYNC_H_

#include <QPointF>
#include <QObject>
#include <QList>

#include <vector>
#include <utility>

class MuratCanvas2dFullView;
class Canvas2dVerticalSync;

class Canvas2dFullSync : public QObject {
	Q_OBJECT
public:
	Canvas2dFullSync(QWidget* parent=0);
	virtual ~Canvas2dFullSync();

	void addCanvas2d(MuratCanvas2dFullView* canvas);
	void remove(MuratCanvas2dFullView* canvas);

	int getCurveOffset();
	int getCurveSize();

	MuratCanvas2dFullView* getExample();
	void setOther(Canvas2dVerticalSync* otherSync);

	bool getDisplayCurves();
	void setDisplayCurves(bool);

	bool getDisplayCursors();
	void setDisplayCursors(bool);

	int getPatchSize();

	void drawRectangles(std::vector<std::pair<QRect, QColor>> rects);

	void clearOverlayRectangles();

public slots:
	void fitImage();
	void zoomIn();
	void zoomOut();
    void simulate_gentle_zoom(double factorY, qreal y_target_scene_pos, MuratCanvas2dFullView* canvas=nullptr);
    void simulate_gentle_zoom(double factorX, double factorY, QPointF target_scene_pos, MuratCanvas2dFullView* canvas=nullptr);
	void simulate_gentle_move(QPointF pos, MuratCanvas2dFullView* canvas=nullptr);
	void simulate_gentle_move(qreal y_target_scene_pos);
	void simulate_scrollbar_value_changed(int value, Qt::Orientation orientation, MuratCanvas2dFullView* canvas=nullptr);

	void simulate_mouse_press_event(QPointF pos, MuratCanvas2dFullView* canvas);

	void setCurveOffset(int value);
	void setCurveSize(int value);
	void setPatchSize(int winsize);

	void centerOn(QPointF);

signals:
    void gentle_zoom(double factorX, double factorY, QPointF target_scene_pos);
	void gentle_move(QPointF pos);
	void scrollbar_value_changed(int value, Qt::Orientation orientation);
    void signalFitImage(qreal ratioX, qreal ratioY);
	void signalZoomIn();
	void signalZoomOut();
	void takePatch(int x, int y);

protected:
	QList<MuratCanvas2dFullView*> canvases;
	Canvas2dVerticalSync* otherSync=nullptr;

	int winSize;
};

#endif /* MURATAPP_SRC_VIEW_CANVAS2D_CANVAS2DFULLSYNC_H_ */

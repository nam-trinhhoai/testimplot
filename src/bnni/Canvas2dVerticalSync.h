#ifndef CANVAS2DVERTICALSYNC_H
#define CANVAS2DVERTICALSYNC_H


#include <QPointF>
#include <QObject>
#include <QList>

class MuratCanvas2dView;
class MuratCanvas2dVerticalView;

class Canvas2dVerticalSync : public QObject {
    Q_OBJECT
public:
    Canvas2dVerticalSync(QWidget* parent=0);
    virtual ~Canvas2dVerticalSync();

    void addCanvas2d(MuratCanvas2dVerticalView* canvas);
    void remove(MuratCanvas2dVerticalView* canvas);

    int getCurveOffset();
    int getCurveSize();

    MuratCanvas2dVerticalView* getExample();
    void forceUpdate();

public slots:
    void fitImage(qreal ratio);
    void zoomIn();
    void zoomOut();
    void simulate_gentle_zoom(double factor, qreal y_target_scene_pos, MuratCanvas2dVerticalView* canvas=nullptr);
    void simulate_gentle_move(qreal y_target_scene_pos, MuratCanvas2dVerticalView* canvas=nullptr);
    void simulate_scrollbar_value_changed(int value, Qt::Orientation orientation, MuratCanvas2dVerticalView* canvas=nullptr);

    void setCurveOffset(int value);
    void setCurveSize(int value);

signals:
    void gentle_zoom(double factor, qreal y_target_scene_pos);
    void scrollbar_value_changed(int value, Qt::Orientation orientation);
    void signalFitImage();
    void signalZoomIn();
    void signalZoomOut();
    void gentle_move(qreal y_target_scene_pos);

protected:
    QList<MuratCanvas2dVerticalView*> canvases;
};

#endif // CANVAS2DVERTICALSYNC_H

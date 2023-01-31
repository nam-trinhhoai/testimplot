#ifndef VIEWPORTMANAGER_H
#define VIEWPORTMANAGER_H

#include <QObject>
#include <QRectF>
#include <QHash>

namespace Qt3DRender {
class QViewport;
}

class ViewportManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(ViewMode previousViewMode READ previousViewMode NOTIFY previousViewModeChanged)
    Q_PROPERTY(ViewMode currentViewMode READ currentViewMode WRITE setCurrentViewMode NOTIFY currentViewModeChanged)
    Q_PROPERTY(Qt3DRender::QViewport *mainViewport READ mainViewport WRITE setMainViewport NOTIFY mainViewportChanged)
    Q_PROPERTY(float freeSliceSubviewPortHeight READ freeSliceSubviewPortHeight WRITE setFreeSliceSubviewPortHeight NOTIFY freeSliceSubviewPortHeightChanged)
    Q_PROPERTY(float freeSliceSubviewPortWidth READ freeSliceSubviewPortWidth WRITE setFreeSliceSubviewPortWidth NOTIFY freeSliceSubviewPortWidthChanged)
	Q_PROPERTY(int nbView3D READ nbView3D WRITE setNbView3D NOTIFY nbView3DChanged)
	Q_PROPERTY(bool modeSplit3D READ modeSplit3D WRITE setModeSplit3D NOTIFY modeSplit3DChanged)
    Q_PROPERTY(Qt3DRender::QViewport *maximizedViewport READ maximizedViewport WRITE setMaximizedViewport NOTIFY maximizedViewportChanged)

public:
    explicit ViewportManager(QObject *parent = nullptr);

    enum ViewMode {
        FreeViewMode = 0,
        View2DIntersectionMode,
        FreeSliceIntersectionMode
    };
    Q_ENUM(ViewMode)

    enum ViewRenderMode {
           	ViewMode3D = 0,
            ViewMode2D,
            ViewModeImage
        };
        Q_ENUM(ViewRenderMode)

    enum ViewportPlaceHolder {
        PlaceHolder_1=0,
        PlaceHolder_2,
        PlaceHolder_3,
        PlaceHolder_4,
		PlaceHolder_5,
		PlaceHolder_6,
		PlaceHolder_7,
		PlaceHolder_8,
    };
    Q_ENUM(ViewportPlaceHolder)

    ViewMode previousViewMode() const;
    ViewMode currentViewMode() const;
    void setCurrentViewMode(ViewMode mode);

    Qt3DRender::QViewport *mainViewport() const;
    void setMainViewport(Qt3DRender::QViewport *viewport);

    Qt3DRender::QViewport *maximizedViewport() const;
    void setMaximizedViewport(Qt3DRender::QViewport *viewport);

    float freeSliceSubviewPortHeight() const;
    void setFreeSliceSubviewPortHeight(float height);

    float freeSliceSubviewPortWidth() const;
    void setFreeSliceSubviewPortWidth(float width);

    int nbView3D() const;
    void setNbView3D(int nbview);

    bool modeSplit3D() const;
    void setModeSplit3D(bool split);

    Q_INVOKABLE void fusionViewport();

    Q_INVOKABLE void removeViewport(Qt3DRender::QViewport *viewport);

    Q_INVOKABLE void addViewport(Qt3DRender::QViewport *viewport, ViewportPlaceHolder placeHolder);
    Q_INVOKABLE void addViewportMulti(ViewRenderMode mode, Qt3DRender::QViewport *viewport,ViewportManager::ViewportPlaceHolder placeHolder, bool modeSplit);
    Q_INVOKABLE void swapViewports(Qt3DRender::QViewport *a, Qt3DRender::QViewport *b);

signals:
    void previousViewModeChanged();
    void currentViewModeChanged();
    void mainViewportChanged();
    void maximizedViewportChanged();
    void freeSliceSubviewPortHeightChanged();
    void freeSliceSubviewPortWidthChanged();
    void nbView3DChanged();

    void modeSplit3DChanged();

private:
    void restoreViewportLayout();
    void placeFreeSliceViewport(Qt3DRender::QViewport *vp, ViewportPlaceHolder placeHolder);

    QRectF computeViewport(int numView ,Qt3DRender::QViewport *viewport);
    ViewMode m_previousViewMode;
    ViewMode m_currentViewMode;
    Qt3DRender::QViewport *m_mainViewport;
    Qt3DRender::QViewport *m_maximizedViewport;
    float m_freeSliceSubviewPortHeight;
    float m_freeSliceSubviewPortWidth;

    QHash<Qt3DRender::QViewport *, ViewportPlaceHolder> m_viewportToPlaceHolder;
    QHash<ViewMode, QRectF> m_mainViewportRectForMode;

    int m_nbView3D = 1;
    int m_maxNbView3D = 4;
    int m_nbView2D = 0;
    int m_nbViewSection = 0;

    bool m_modeSplit3D = true;
};

#endif // VIEWPORTMANAGER_H

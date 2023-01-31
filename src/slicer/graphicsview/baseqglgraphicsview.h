#ifndef BaseQGLGraphicsView_H
#define BaseQGLGraphicsView_H
#include <QMouseEvent>
#include <QGraphicsView>
#include <QMenu>

class AbstractOverlayPainter {
public:
	virtual ~AbstractOverlayPainter();
	/*
	 * rect contains the painting area bounding box
	 */
	virtual void paintOverlay(QPainter* painter, const QRectF& rect) = 0;
};


class BaseQGLGraphicsView: public QGraphicsView {
Q_OBJECT
public:
	BaseQGLGraphicsView(QWidget *parent = 0);
	virtual ~BaseQGLGraphicsView() {
	}
	virtual std::pair<float, float> resetZoom(void);
	virtual std::pair<float, float> setVisibleRect(const QRectF &bbox);

	void lockZoom(bool lock);

	bool zoomLocked() const {
		return m_zoomLocked;
	}
	static void resetScale(QGraphicsView *view);

    void addOverlayPainter(AbstractOverlayPainter* painter);
    void removeOverlayPainter(AbstractOverlayPainter* painter);




signals:
	void scaleChanged(double sx, double sy);

	void mouseMoved(double worldX, double worldY, Qt::MouseButton button,
			Qt::KeyboardModifiers keys);
	void mousePressed(double worldX, double worldY, Qt::MouseButton button,
			Qt::KeyboardModifiers keys);
	void mouseRelease(double worldX, double worldY, Qt::MouseButton button,
			Qt::KeyboardModifiers keys);
	void mouseDoubleClick(double worldX, double worldY, Qt::MouseButton button,
                        Qt::KeyboardModifiers keys);
	void contextMenu(double worldX, double worldY, QContextMenuEvent::Reason reason,
			QMenu& menu); // only use direct connection for the QMenu object to be valid
protected:
	virtual void mouseMoveEvent(QMouseEvent *event) override;
	virtual void mousePressEvent(QMouseEvent *event) override;
	virtual void mouseReleaseEvent(QMouseEvent *event) override;
	virtual void mouseDoubleClickEvent(QMouseEvent *event) override;
	virtual void contextMenuEvent(QContextMenuEvent *event) override;

protected:
	void applyScale(double sx, double sy);
	void drawForeground( QPainter* painter, const QRectF& rect ) override;
	QRectF getViewportRect( ) const;

private:
	bool m_zoomLocked;
	QList<AbstractOverlayPainter*> m_overlayPainters;
};

#endif

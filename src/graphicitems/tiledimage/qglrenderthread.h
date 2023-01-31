#ifndef QGLRenderThread_h_
#define QGLRenderThread_h_

#include "qglrenderqueue.h"
#include "qgltile.h"
#include <QThread>
#include <QHash>
#include <QImage>
#include <QMutex>
#include <QList>
#include <QMap>

class QGLAbstractTiledImage;

class QGLRenderThread : public QThread
{
	Q_OBJECT

public:
	QGLRenderThread(QGLAbstractTiledImage *image,QObject *parent);
	~QGLRenderThread();

	void requestTile(const QGLTileCoord &coords);

	void startService();
	void stopService(unsigned long waitTime=ULONG_MAX);
	void reset(void);

protected:
	virtual void run(void);

	bool m_stopped;
	QGLAbstractTiledImage * m_image;

	QWaitCondition m_queueNotEmpty;
	QGLRenderQueue m_eventQueue;
};
#endif

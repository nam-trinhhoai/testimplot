#include "qglrenderthread.h"
#include <QBuffer>
#include <QDebug>
#include <QTextStream>
#include <limits.h>

#include "qglabstracttiledimage.h"

QGLRenderThread::QGLRenderThread(QGLAbstractTiledImage *image, QObject *parent) :
		QThread(parent) {
	m_image = image;
	m_stopped = true;
}

QGLRenderThread::~QGLRenderThread() {
	stopService(5000);
}

void QGLRenderThread::requestTile(const QGLTileCoord &coords) {
	m_eventQueue.push(coords);
}

void QGLRenderThread::startService() {
	m_stopped = false;
	reset();
	start();
}

void QGLRenderThread::stopService(unsigned long waitTime) {
	m_stopped = true;

	//We push a void data to force dequeue
	QGLTileCoord coords;
	m_eventQueue.push(coords);

	wait(waitTime);
	m_eventQueue.clear();
}

void QGLRenderThread::reset(void) {
	m_eventQueue.clear();
}

//virtual
void QGLRenderThread::run(void) {
	forever {
		QGLTileCoord item;
		if (m_eventQueue.waitAndPop(item)) {
			//When stopping we dequeue one item
			if (m_stopped)
				break;
			m_image->addImageTileToCache(item);
		}
	}
}


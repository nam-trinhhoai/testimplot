#ifndef QGLRenderQueue_h_
#define QGLRenderQueue_h_

#include <QMutex>
#include <QWaitCondition>
#include <QQueue>
#include <iostream>

#include "qgltile.h"

class QGLRenderQueue : protected QQueue<QGLTileCoord>
{
	typedef QQueue<QGLTileCoord> Container;
public:
	void push(const QGLTileCoord& newItem)
	{
		QMutexLocker locker(&m_queueLock);
		Container::push_back(newItem);
		m_queueNotEmpty.wakeOne();
	}
	bool empty(void) const
	{
		QMutexLocker locker(&m_queueLock);
		return Container::empty();
	}
	bool tryPop(QGLTileCoord& item)
	{
		QMutexLocker locker(&m_queueLock);
		bool popped = false;
		if(!Container::empty()) {
			item = Container::dequeue();
			popped = true;
		}
		return popped;
	}
	bool waitAndPop(QGLTileCoord& item)
	{
		QMutexLocker locker(&m_queueLock);
		bool popped = false;
		while(Container::empty()) {
			m_queueNotEmpty.wait(&m_queueLock);
		}
		if(!Container::empty()) {
			item = Container::dequeue();
			popped = true;
		}
		return popped;
	}
	void clear(void)
	{
		QMutexLocker locker(&m_queueLock);
		while(!Container::empty()) {
			Container::dequeue();
		}
		Container::clear();
	}
private:
	mutable QMutex m_queueLock;
	QWaitCondition m_queueNotEmpty;
	bool closed;
};


#endif


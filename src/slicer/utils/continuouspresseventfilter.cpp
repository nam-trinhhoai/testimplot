#include "continuouspresseventfilter.h"

#include <QDebug>

ContinuousPressEventFilter::ContinuousPressEventFilter(QList<int> filterKeys, QObject* parent) : QObject(parent) {
	m_filterKeys = filterKeys;
}

ContinuousPressEventFilter::~ContinuousPressEventFilter() {}

bool ContinuousPressEventFilter::eventFilter(QObject* object, QEvent* event) {
	QKeyEvent *ev = dynamic_cast<QKeyEvent *>(event);
	if ( event->type() == QEvent::KeyPress) {
		if (m_filterKeys.contains(ev->key())) {
//			qDebug() << "CPEF : detect press key" << ev->key();
			// special processing for key press

			QMap<int, std::shared_ptr<QTimer>>::key_iterator it = m_filterReleaseKeyMap.keyBegin();
			while (it!=m_filterReleaseKeyMap.keyEnd() && (*it)!=ev->key()) {
				it++;
			}
			if (it==m_filterReleaseKeyMap.keyEnd()) {
				std::shared_ptr<QTimer> timerPtr(nullptr);
				m_filterReleaseKeyMap.insert(ev->key(), timerPtr);
//				qDebug() << "CPEF : emit press for key" << ev->key();
				emit keyPressSignal(ev->key());
			} else if (m_filterReleaseKeyMap[ev->key()].get()!=nullptr) {
//				qDebug() << "CPEF : no emit for key" << ev->key();
				m_filterReleaseKeyMap[ev->key()]->stop();
			}
			return true; // eat event
			/*if (!ev->isAutoRepeat()) {
				qDebug() << "CPEF : emit press for key" << ev->key();
				emit keyPressSignal(ev->key());
				return true;
			} else {
				qDebug() << "CPEF : no emit for key" << ev->key();
				return false;
			}*/
		}
	} else if ( event->type() == QEvent::KeyRelease) {
		if (m_filterKeys.contains(ev->key())) {
			// special processing for key press
			QKeyEvent *ev = dynamic_cast<QKeyEvent *>(event);
//			qDebug() << "CPEF : detect release for key" << ev->key();
			/*if (!ev->isAutoRepeat()) {
				qDebug() << "CPEF : emit release for key" << ev->key();
				emit keyReleaseSignal(ev->key());
				return true;
			} else {
				qDebug() << "CPEF : no emit for key" << ev->key();
				return false;
			}*/
			//emit keyReleaseSignal(ev);

			QMap<int, std::shared_ptr<QTimer>>::key_iterator it = m_filterReleaseKeyMap.keyBegin();
			while (it!=m_filterReleaseKeyMap.keyEnd() && (*it)!=ev->key()) {
				it++;
			}
			std::shared_ptr<QTimer> timerPtr;
			bool isUnset = false;
			if (it!=m_filterReleaseKeyMap.keyEnd()) {
				timerPtr = m_filterReleaseKeyMap[ev->key()];
				isUnset = timerPtr.get()==nullptr;
			}
			if (it==m_filterReleaseKeyMap.keyEnd() || isUnset) {
				timerPtr.reset(new QTimer);
				timerPtr->setSingleShot(true);
				if (it==m_filterReleaseKeyMap.keyEnd()) {
					m_filterReleaseKeyMap.insert(ev->key(), timerPtr);
				} else {
					m_filterReleaseKeyMap[ev->key()] = timerPtr;
				}
				int key = ev->key();
//				qDebug() << "CPEF : create timer release for key" << ev->key();
				QObject::connect(timerPtr.get(), &QTimer::timeout, [this, key]() {
					QMap<int, std::shared_ptr<QTimer>>::iterator it = m_filterReleaseKeyMap.begin();
					while (it!=m_filterReleaseKeyMap.end() && it.key()!=key) {
						it++;
					}
					if (it!=m_filterReleaseKeyMap.end()) {
						m_filterReleaseKeyMap.erase(it);
//						qDebug() << "CPEF : emit release for key" << key;
						emit keyReleaseSignal(key);
					}
				});
			}
//			qDebug() << "CPEF : reset timer release for key";
			timerPtr->start(200);

		}
	}
	// standard event processing
	return false;
}


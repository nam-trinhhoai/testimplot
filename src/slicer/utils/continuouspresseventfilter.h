#ifndef CONTINUOUSPRESSEVENTFILTER_H
#define CONTINUOUSPRESSEVENTFILTER_H

#include <QObject>
#include <QTimer>
#include <QEvent>
#include <QKeyEvent>
#include <memory>

class ContinuousPressEventFilter : public QObject {
	Q_OBJECT
public:
	ContinuousPressEventFilter(QList<int> filterKeys, QObject* parent=0);
	~ContinuousPressEventFilter();

signals:
	void keyPressSignal(int key);
	void keyReleaseSignal(int key);

protected:
	virtual bool eventFilter(QObject* object, QEvent* event) override;

private:
	QMap<int, std::shared_ptr<QTimer>> m_filterReleaseKeyMap;
	QList<int> m_filterKeys;
};

#endif

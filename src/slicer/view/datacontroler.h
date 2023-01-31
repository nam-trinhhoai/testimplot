#ifndef DataControler_H
#define DataControler_H
#include <QUuid>
#include <QObject>

class DataControler : public QObject{
	  Q_OBJECT
public:
	DataControler(QObject *parent);
	virtual ~DataControler();

	QObject * provider() const;
	virtual QUuid dataID() const=0;
protected:
	QObject *m_provider;
};

#endif

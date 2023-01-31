#ifndef IGraphicRepFactory_H
#define IGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

class AbstractInnerView;
class AbstractGraphicRep;

class IGraphicRepFactory: public QObject {
Q_OBJECT
public:
	IGraphicRepFactory(QObject *parent = 0);
	virtual ~IGraphicRepFactory();

	//Retreive the child of this data
	virtual QList<IGraphicRepFactory*> childReps(ViewType type,AbstractInnerView *parent);
	virtual AbstractGraphicRep* rep(ViewType type, AbstractInnerView *parent)=0;

signals:
	void childAdded(IGraphicRepFactory *child);
	void childRemoved(IGraphicRepFactory *child);
};

#endif

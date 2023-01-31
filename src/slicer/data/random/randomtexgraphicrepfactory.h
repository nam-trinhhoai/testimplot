#ifndef RandomTexGraphicRepFactory_H
#define RandomTexGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"
#include "randomtexdataset.h"

#include "igraphicrepfactory.h"
class RandomTexDataset;


class RandomTexGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	RandomTexGraphicRepFactory(RandomTexDataset * data);
	virtual ~RandomTexGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;
	virtual QList<IGraphicRepFactory*> childReps(ViewType type,AbstractInnerView *parent) override;



private:
	RandomTexDataset * m_data;

};

#endif

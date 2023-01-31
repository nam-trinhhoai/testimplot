#ifndef RandomGraphicRepFactory_H
#define RandomGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"
#include "randomdataset.h"
#include "randomtexdataset.h"

#include "igraphicrepfactory.h"
class RandomDataset;
class RandomTexDataset;


class RandomGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	RandomGraphicRepFactory(RandomDataset * data);
	virtual ~RandomGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;
	virtual QList<IGraphicRepFactory*> childReps(ViewType type,AbstractInnerView *parent) override;


	private slots:
		void datasetAdded(RandomTexDataset *dataset);
		void datasetRemoved(RandomTexDataset *dataset);

private:
	RandomDataset * m_data;

};

#endif

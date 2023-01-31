#ifndef ScalarCompurationOnDatasetGraphicRepFactory_H
#define ScalarCompurationOnDatasetGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class ScalarComputationOnDataset;
class AbstractInnerView;

class ScalarComputationOnDatasetGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	ScalarComputationOnDatasetGraphicRepFactory(ScalarComputationOnDataset * data);
	virtual ~ScalarComputationOnDatasetGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;

private:
	ScalarComputationOnDataset* m_data;

};

#endif

#ifndef ComputationOperatorDatasetGraphicRepFactory_H
#define ComputationOperatorDatasetGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class ComputationOperatorDataset;
class AbstractInnerView;

class ComputationOperatorDatasetGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	ComputationOperatorDatasetGraphicRepFactory(ComputationOperatorDataset * data);
	virtual ~ComputationOperatorDatasetGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;

private:
	ComputationOperatorDataset* m_data;

};

#endif

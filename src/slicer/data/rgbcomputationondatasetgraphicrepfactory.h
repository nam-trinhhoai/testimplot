#ifndef RgbCompurationOnDatasetGraphicRepFactory_H
#define RgbCompurationOnDatasetGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class RgbComputationOnDataset;
class AbstractInnerView;

class RgbComputationOnDatasetGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	RgbComputationOnDatasetGraphicRepFactory(RgbComputationOnDataset * data);
	virtual ~RgbComputationOnDatasetGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;

private:
	RgbComputationOnDataset* m_data;

};

#endif

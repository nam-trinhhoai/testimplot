#ifndef FixedRGBLayersFromDatasetAndCubeGraphicRepFactory_H
#define FixedRGBLayersFromDatasetAndCubeGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class FixedRGBLayersFromDatasetAndCube;

class FixedRGBLayersFromDatasetAndCubeGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	FixedRGBLayersFromDatasetAndCubeGraphicRepFactory(FixedRGBLayersFromDatasetAndCube * data);
	virtual ~FixedRGBLayersFromDatasetAndCubeGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
	virtual QList<IGraphicRepFactory *> childReps(ViewType type,AbstractInnerView * parent)  override;

private:
	FixedRGBLayersFromDatasetAndCube * m_data;

};

#endif

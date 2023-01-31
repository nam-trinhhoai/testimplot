#ifndef FixedLayersFromDatasetAndCubeGraphicRepFactory_H
#define FixedLayersFromDatasetAndCubeGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class FixedLayersFromDatasetAndCube;

class FixedLayersFromDatasetAndCubeGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	FixedLayersFromDatasetAndCubeGraphicRepFactory(FixedLayersFromDatasetAndCube * data);
	virtual ~FixedLayersFromDatasetAndCubeGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	FixedLayersFromDatasetAndCube * m_data;

};

#endif

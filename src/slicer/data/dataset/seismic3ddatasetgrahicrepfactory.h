#ifndef Seismic3DDatasetGraphicRepFactory_H
#define Seismic3DDatasetGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"
#include "sliceutils.h"

class Seismic3DAbstractDataset;

class Seismic3DDatasetGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	Seismic3DDatasetGraphicRepFactory(Seismic3DAbstractDataset * data);
	virtual ~Seismic3DDatasetGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	Seismic3DAbstractDataset * m_data;

};

#endif

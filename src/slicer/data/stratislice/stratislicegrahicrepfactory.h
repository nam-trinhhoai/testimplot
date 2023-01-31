#ifndef StratiSliceGraphicRepFactory_H
#define StratiSliceGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class StratiSlice;

class StratiSliceGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	StratiSliceGraphicRepFactory(StratiSlice * data);
	virtual ~StratiSliceGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
	virtual QList<IGraphicRepFactory *> childReps(ViewType type,AbstractInnerView * parent)  override;
private:
	StratiSlice * m_data;

};

#endif

#ifndef StratiSliceRGBAttributeGraphicRepFactory_H
#define StratiSliceRGBAttributeGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"


class RGBStratiSliceAttribute;

class StratiSliceRGBAttributeGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	StratiSliceRGBAttributeGraphicRepFactory(RGBStratiSliceAttribute * data);
	virtual ~StratiSliceRGBAttributeGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	RGBStratiSliceAttribute * m_data;

};

#endif

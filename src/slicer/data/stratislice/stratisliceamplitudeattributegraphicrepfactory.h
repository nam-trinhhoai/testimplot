#ifndef StratiSliceAmplitudeAttributeGraphicRepFactory_H
#define StratiSliceAmplitudeAttributeGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class AmplitudeStratiSliceAttribute;

class StratiSliceAmplitudeAttributeGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	StratiSliceAmplitudeAttributeGraphicRepFactory(AmplitudeStratiSliceAttribute * data);
	virtual ~StratiSliceAmplitudeAttributeGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	AmplitudeStratiSliceAttribute * m_data;

};

#endif

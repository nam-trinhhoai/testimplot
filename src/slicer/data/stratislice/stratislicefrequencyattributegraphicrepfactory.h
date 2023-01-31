#ifndef StratiSliceFrequencyAttributeGraphicRepFactory_H
#define StratiSliceFrequencyAttributeGraphicRepFactory_H

#include <QObject>
#include "viewutils.h"
#include "igraphicrepfactory.h"

class FrequencyStratiSliceAttribute;

class StratiSliceFrequencyAttributeGraphicRepFactory:public IGraphicRepFactory{
	Q_OBJECT
public:
	StratiSliceFrequencyAttributeGraphicRepFactory(FrequencyStratiSliceAttribute * data);
	virtual ~StratiSliceFrequencyAttributeGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;
private:
	FrequencyStratiSliceAttribute * m_data;

};

#endif
